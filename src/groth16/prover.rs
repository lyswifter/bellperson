use std::ops::{AddAssign, Mul, MulAssign};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, Duration};

use digest::crypto_common::KeyInit;
use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve};
use pairing::MultiMillerLoop;
use rand::Rng;
use rand_core::RngCore;
use rayon::prelude::*;

use super::{ParameterSource, Proof};
use crate::domain::EvaluationDomain;
use crate::gpu::{GpuName, LockedFftKernel, LockedMultiexpKernel, CpuGpuMultiexpKernel};
use crate::multiexp::multiexp;
use crate::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable, BELLMAN_VERSION,
};
use ec_gpu_gen::multiexp_cpu::{DensityTracker, FullDensity};
use ec_gpu_gen::threadpool::{Worker, THREAD_POOL};
#[cfg(any(feature = "cuda", feature = "opencl"))]
use log::trace;
use log::{debug, info, warn};

#[cfg(any(feature = "cuda", feature = "opencl"))]
use crate::gpu::PriorityLock;

use pairing::Engine;
use std::thread::sleep;
use crate::groth16::verify_proof;
use crate::groth16::PreparedVerifyingKey;
use cuda_builder_ffi::params::*;

static RNNING_SYN_THREAD_NUM: AtomicUsize = AtomicUsize::new(0);
static RNNING_GPU_THREAD_NUM: AtomicUsize = AtomicUsize::new(0);
static THIS_THREAD_NUM: AtomicUsize = AtomicUsize::new(1);
static RNNING_THREAD_NUM: AtomicUsize = AtomicUsize::new(0);
static CACHE_BUILDING: AtomicUsize = AtomicUsize::new(0);

static FFT_THRESH: usize = 9000;
static FFT_N_VALUE: usize = 27; //2^27 = 134217728

struct ProvingAssignment<Scalar: PrimeField> {
    // Density of queries
    a_aux_density: DensityTracker,
    b_input_density: DensityTracker,
    b_aux_density: DensityTracker,

    // Evaluations of A, B, C polynomials
    a: Vec<Scalar>,
    b: Vec<Scalar>,
    c: Vec<Scalar>,

    // Assignments of variables
    input_assignment: Vec<Scalar>,
    aux_assignment: Vec<Scalar>,
}
use std::fmt;

impl<Scalar: PrimeField> fmt::Debug for ProvingAssignment<Scalar> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("ProvingAssignment")
            .field("a_aux_density", &self.a_aux_density)
            .field("b_input_density", &self.b_input_density)
            .field("b_aux_density", &self.b_aux_density)
            .field(
                "a",
                &self
                    .a
                    .iter()
                    .map(|v| format!("Fr({:?})", v))
                    .collect::<Vec<_>>(),
            )
            .field(
                "b",
                &self
                    .b
                    .iter()
                    .map(|v| format!("Fr({:?})", v))
                    .collect::<Vec<_>>(),
            )
            .field(
                "c",
                &self
                    .c
                    .iter()
                    .map(|v| format!("Fr({:?})", v))
                    .collect::<Vec<_>>(),
            )
            .field("input_assignment", &self.input_assignment)
            .field("aux_assignment", &self.aux_assignment)
            .finish()
    }
}

impl<Scalar: PrimeField> PartialEq for ProvingAssignment<Scalar> {
    fn eq(&self, other: &ProvingAssignment<Scalar>) -> bool {
        self.a_aux_density == other.a_aux_density
            && self.b_input_density == other.b_input_density
            && self.b_aux_density == other.b_aux_density
            && self.a == other.a
            && self.b == other.b
            && self.c == other.c
            && self.input_assignment == other.input_assignment
            && self.aux_assignment == other.aux_assignment
    }
}

impl<Scalar: PrimeField> ConstraintSystem<Scalar> for ProvingAssignment<Scalar> {
    type Root = Self;

    fn new() -> Self {
        Self {
            a_aux_density: DensityTracker::new(),
            b_input_density: DensityTracker::new(),
            b_aux_density: DensityTracker::new(),
            a: vec![],
            b: vec![],
            c: vec![],
            input_assignment: vec![],
            aux_assignment: vec![],
        }
    }

    fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.aux_assignment.push(f()?);
        self.a_aux_density.add_element();
        self.b_aux_density.add_element();

        Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.input_assignment.push(f()?);
        self.b_input_density.add_element();

        Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
        let a = a(LinearCombination::zero());
        let b = b(LinearCombination::zero());
        let c = c(LinearCombination::zero());

        let input_assignment = &self.input_assignment;
        let aux_assignment = &self.aux_assignment;
        let a_aux_density = &mut self.a_aux_density;
        let b_input_density = &mut self.b_input_density;
        let b_aux_density = &mut self.b_aux_density;

        let a_res = a.eval(
            // Inputs have full density in the A query
            // because there are constraints of the
            // form x * 0 = 0 for each input.
            None,
            Some(a_aux_density),
            input_assignment,
            aux_assignment,
        );

        let b_res = b.eval(
            Some(b_input_density),
            Some(b_aux_density),
            input_assignment,
            aux_assignment,
        );

        let c_res = c.eval(
            // There is no C polynomial query,
            // though there is an (beta)A + (alpha)B + C
            // query for all aux variables.
            // However, that query has full density.
            None,
            None,
            input_assignment,
            aux_assignment,
        );

        self.a.push(a_res);
        self.b.push(b_res);
        self.c.push(c_res);
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn is_extensible() -> bool {
        true
    }

    fn extend(&mut self, other: Self) {
        self.a_aux_density.extend(other.a_aux_density, false);
        self.b_input_density.extend(other.b_input_density, true);
        self.b_aux_density.extend(other.b_aux_density, false);

        self.a.extend(other.a);
        self.b.extend(other.b);
        self.c.extend(other.c);

        self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
        self.aux_assignment.extend(other.aux_assignment);
    }
}

pub fn create_random_proof_batch_priority<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    R: RngCore,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    let r_s = (0..circuits.len())
        .map(|_| E::Fr::random(&mut *rng))
        .collect();
    let s_s = (0..circuits.len())
        .map(|_| E::Fr::random(&mut *rng))
        .collect();

    create_proof_batch_priority::<E, C, P>(circuits, params, r_s, s_s, priority)
}

/// creates a batch of proofs where the randomization vector is set to zero.
/// This allows for optimization of proving.
pub fn create_proof_batch_priority_nonzk<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    create_proof_batch_priority_inner(circuits, params, None, priority)
}

/// creates a batch of proofs where the randomization vector is already
/// predefined
#[allow(clippy::needless_collect)]
pub fn create_proof_batch_priority<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    r_s: Vec<E::Fr>,
    s_s: Vec<E::Fr>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    create_proof_batch_priority_inner(circuits, params, Some((r_s, s_s)), priority)
}

#[allow(clippy::type_complexity)]
#[allow(clippy::needless_collect)]
fn create_proof_batch_priority_inner<E, C, P: ParameterSource<E>>(
    circuits: Vec<C>,
    params: P,
    randomization: Option<(Vec<E::Fr>, Vec<E::Fr>)>,
    priority: bool,
) -> Result<Vec<Proof<E>>, SynthesisError>
where
    E: MultiMillerLoop,
    C: Circuit<E::Fr> + Send,
    E::Fr: GpuName,
    E::G1Affine: GpuName,
    E::G2Affine: GpuName,
{
    info!("Bellperson {} is being used!", BELLMAN_VERSION);

    let (start, mut provers, input_assignments, aux_assignments) =
        synthesize_circuits_batch(circuits)?;

    let worker = Worker::new();
    let input_len = input_assignments[0].len();
    let vk = params.get_vk(input_len)?.clone();
    let n = provers[0].a.len();
    let a_aux_density_total = provers[0].a_aux_density.get_total_density();
    let b_input_density_total = provers[0].b_input_density.get_total_density();
    let b_aux_density_total = provers[0].b_aux_density.get_total_density();
    let aux_assignment_len = provers[0].aux_assignment.len();
    let num_circuits = provers.len();

    let zk = randomization.is_some();
    let (r_s, s_s) = randomization.unwrap_or((
        vec![E::Fr::zero(); num_circuits],
        vec![E::Fr::zero(); num_circuits],
    ));

    // Make sure all circuits have the same input len.
    for prover in &provers {
        assert_eq!(
            prover.a.len(),
            n,
            "only equaly sized circuits are supported"
        );
        debug_assert_eq!(
            a_aux_density_total,
            prover.a_aux_density.get_total_density(),
            "only identical circuits are supported"
        );
        debug_assert_eq!(
            b_input_density_total,
            prover.b_input_density.get_total_density(),
            "only identical circuits are supported"
        );
        debug_assert_eq!(
            b_aux_density_total,
            prover.b_aux_density.get_total_density(),
            "only identical circuits are supported"
        );
    }

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    let prio_lock = if priority {
        trace!("acquiring priority lock");
        Some(PriorityLock::lock())
    } else {
        None
    };

    let mut a_s = Vec::with_capacity(num_circuits);
    let mut params_h = None;
    let worker = &worker;
    let provers_ref = &mut provers;
    let params = &params;

    THREAD_POOL.scoped(|s| -> Result<(), SynthesisError> {
        let params_h = &mut params_h;
        s.execute(move || {
            debug!("get h");
            *params_h = Some(params.get_h(n));
        });

        let mut fft_kern = Some(LockedFftKernel::new(priority, None));
        for prover in provers_ref {
            a_s.push(execute_fft(worker, prover, &mut fft_kern)?);
        }
        Ok(())
    })?;

    let mut multiexp_g1_kern = LockedMultiexpKernel::<E::G1Affine>::new(priority, None);
    let params_h = params_h.unwrap()?;

    let mut h_s = Vec::with_capacity(num_circuits);
    let mut params_l = None;

    THREAD_POOL.scoped(|s| {
        let params_l = &mut params_l;
        s.execute(move || {
            debug!("get l");
            *params_l = Some(params.get_l(aux_assignment_len));
        });

        debug!("multiexp h");
        for a in a_s.into_iter() {
            h_s.push(multiexp(
                worker,
                params_h.clone(),
                FullDensity,
                a,
                &mut multiexp_g1_kern,
            ));
        }
    });

    let params_l = params_l.unwrap()?;

    let mut l_s = Vec::with_capacity(num_circuits);
    let mut params_a = None;
    let mut params_b_g1 = None;
    let mut params_b_g2 = None;
    let a_aux_density_total = provers[0].a_aux_density.get_total_density();
    let b_input_density_total = provers[0].b_input_density.get_total_density();
    let b_aux_density_total = provers[0].b_aux_density.get_total_density();

    THREAD_POOL.scoped(|s| {
        let params_a = &mut params_a;
        let params_b_g1 = &mut params_b_g1;
        let params_b_g2 = &mut params_b_g2;
        s.execute(move || {
            debug!("get_a b_g1 b_g2");
            *params_a = Some(params.get_a(input_len, a_aux_density_total));
            if zk {
                *params_b_g1 = Some(params.get_b_g1(b_input_density_total, b_aux_density_total));
            }
            *params_b_g2 = Some(params.get_b_g2(b_input_density_total, b_aux_density_total));
        });

        debug!("multiexp l");
        for aux in aux_assignments.iter() {
            l_s.push(multiexp(
                worker,
                params_l.clone(),
                FullDensity,
                aux.clone(),
                &mut multiexp_g1_kern,
            ));
        }
    });

    debug!("get a b_g1");
    let (a_inputs_source, a_aux_source) = params_a.unwrap()?;
    let params_b_g1_opt = params_b_g1.transpose()?;

    let densities = provers
        .iter_mut()
        .map(|prover| {
            let a_aux_density = std::mem::take(&mut prover.a_aux_density);
            let b_input_density = std::mem::take(&mut prover.b_input_density);
            let b_aux_density = std::mem::take(&mut prover.b_aux_density);
            (
                Arc::new(a_aux_density),
                Arc::new(b_input_density),
                Arc::new(b_aux_density),
            )
        })
        .collect::<Vec<_>>();
    drop(provers);

    debug!("multiexp a b_g1");
    let inputs_g1 = input_assignments
        .iter()
        .zip(aux_assignments.iter())
        .zip(densities.iter())
        .map(
            |(
                (input_assignment, aux_assignment),
                (a_aux_density, b_input_density, b_aux_density),
            )| {
                let a_inputs = multiexp(
                    worker,
                    a_inputs_source.clone(),
                    FullDensity,
                    input_assignment.clone(),
                    &mut multiexp_g1_kern,
                );

                let a_aux = multiexp(
                    worker,
                    a_aux_source.clone(),
                    a_aux_density.clone(),
                    aux_assignment.clone(),
                    &mut multiexp_g1_kern,
                );

                let b_g1_inputs_aux_opt =
                    params_b_g1_opt
                        .as_ref()
                        .map(|(b_g1_inputs_source, b_g1_aux_source)| {
                            (
                                multiexp(
                                    worker,
                                    b_g1_inputs_source.clone(),
                                    b_input_density.clone(),
                                    input_assignment.clone(),
                                    &mut multiexp_g1_kern,
                                ),
                                multiexp(
                                    worker,
                                    b_g1_aux_source.clone(),
                                    b_aux_density.clone(),
                                    aux_assignment.clone(),
                                    &mut multiexp_g1_kern,
                                ),
                            )
                        });

                (a_inputs, a_aux, b_g1_inputs_aux_opt)
            },
        )
        .collect::<Vec<_>>();
    drop(multiexp_g1_kern);
    drop(a_inputs_source);
    drop(a_aux_source);
    drop(params_b_g1_opt);

    // The multiexp kernel for G1 can only be initiated after the kernel for G1 was dropped. Else
    // it would block, trying to acquire the GPU lock.
    let mut multiexp_g2_kern = LockedMultiexpKernel::<E::G2Affine>::new(priority, None);

    debug!("get b_g2");
    let (b_g2_inputs_source, b_g2_aux_source) = params_b_g2.unwrap()?;

    debug!("multiexp b_g2");
    let inputs_g2 = input_assignments
        .iter()
        .zip(aux_assignments.iter())
        .zip(densities.iter())
        .map(
            |((input_assignment, aux_assignment), (_, b_input_density, b_aux_density))| {
                let b_g2_inputs = multiexp(
                    worker,
                    b_g2_inputs_source.clone(),
                    b_input_density.clone(),
                    input_assignment.clone(),
                    &mut multiexp_g2_kern,
                );
                let b_g2_aux = multiexp(
                    worker,
                    b_g2_aux_source.clone(),
                    b_aux_density.clone(),
                    aux_assignment.clone(),
                    &mut multiexp_g2_kern,
                );

                (b_g2_inputs, b_g2_aux)
            },
        )
        .collect::<Vec<_>>();
    drop(multiexp_g2_kern);
    drop(densities);
    drop(b_g2_inputs_source);
    drop(b_g2_aux_source);

    debug!("proofs");
    let proofs = h_s
        .into_iter()
        .zip(l_s.into_iter())
        .zip(inputs_g1.into_iter())
        .zip(inputs_g2.into_iter())
        .zip(r_s.into_iter())
        .zip(s_s.into_iter())
        .map(
            |(
                ((((h, l), (a_inputs, a_aux, b_g1_inputs_aux_opt)), (b_g2_inputs, b_g2_aux)), r),
                s,
            )| {
                if (vk.delta_g1.is_identity() | vk.delta_g2.is_identity()).into() {
                    // If this element is zero, someone is trying to perform a
                    // subversion-CRS attack.
                    return Err(SynthesisError::UnexpectedIdentity);
                }

                let mut g_a = vk.delta_g1.mul(r);
                g_a.add_assign(&vk.alpha_g1);
                let mut g_b = vk.delta_g2.mul(s);
                g_b.add_assign(&vk.beta_g2);
                let mut a_answer = a_inputs.wait()?;
                a_answer.add_assign(&a_aux.wait()?);
                g_a.add_assign(&a_answer);
                a_answer.mul_assign(s);
                let mut g_c = a_answer;

                let mut b2_answer = b_g2_inputs.wait()?;
                b2_answer.add_assign(&b_g2_aux.wait()?);

                g_b.add_assign(&b2_answer);

                if let Some((b_g1_inputs, b_g1_aux)) = b_g1_inputs_aux_opt {
                    let mut b1_answer = b_g1_inputs.wait()?;
                    b1_answer.add_assign(&b_g1_aux.wait()?);
                    b1_answer.mul_assign(r);
                    g_c.add_assign(&b1_answer);
                    let mut rs = r;
                    rs.mul_assign(&s);
                    g_c.add_assign(vk.delta_g1.mul(rs));
                    g_c.add_assign(&vk.alpha_g1.mul(s));
                    g_c.add_assign(&vk.beta_g1.mul(r));
                }

                g_c.add_assign(&h.wait()?);
                g_c.add_assign(&l.wait()?);

                Ok(Proof {
                    a: g_a.to_affine(),
                    b: g_b.to_affine(),
                    c: g_c.to_affine(),
                })
            },
        )
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    {
        trace!("dropping priority lock");
        drop(prio_lock);
    }

    let proof_time = start.elapsed();
    info!("prover time: {:?}", proof_time);

    Ok(proofs)
}

fn execute_fft<F>(
    worker: &Worker,
    prover: &mut ProvingAssignment<F>,
    fft_kern: &mut Option<LockedFftKernel<F>>,
) -> Result<Arc<Vec<F::Repr>>, SynthesisError>
where
    F: PrimeField + GpuName,
{
    let mut a = EvaluationDomain::from_coeffs(std::mem::take(&mut prover.a))?;
    let mut b = EvaluationDomain::from_coeffs(std::mem::take(&mut prover.b))?;
    let mut c = EvaluationDomain::from_coeffs(std::mem::take(&mut prover.c))?;

    EvaluationDomain::ifft_many(&mut [&mut a, &mut b, &mut c], worker, fft_kern)?;
    EvaluationDomain::coset_fft_many(&mut [&mut a, &mut b, &mut c], worker, fft_kern)?;

    a.mul_assign(worker, &b);
    drop(b);
    a.sub_assign(worker, &c);
    drop(c);

    a.divide_by_z_on_coset(worker);
    a.icoset_fft(worker, fft_kern)?;

    let a = a.into_coeffs();
    let a_len = a.len() - 1;
    let a = a
        .into_par_iter()
        .take(a_len)
        .map(|s| s.to_repr())
        .collect::<Vec<_>>();
    Ok(Arc::new(a))
}

#[allow(clippy::type_complexity)]
fn synthesize_circuits_batch<Scalar, C>(
    circuits: Vec<C>,
) -> Result<
    (
        Instant,
        std::vec::Vec<ProvingAssignment<Scalar>>,
        std::vec::Vec<std::sync::Arc<std::vec::Vec<<Scalar as PrimeField>::Repr>>>,
        std::vec::Vec<std::sync::Arc<std::vec::Vec<<Scalar as PrimeField>::Repr>>>,
    ),
    SynthesisError,
>
where
    Scalar: PrimeField,
    C: Circuit<Scalar> + Send,
{
    let start = Instant::now();
    let mut provers = circuits
        .into_par_iter()
        .map(|circuit| -> Result<_, SynthesisError> {
            let mut prover = ProvingAssignment::new();

            prover.alloc_input(|| "", || Ok(Scalar::one()))?;

            circuit.synthesize(&mut prover)?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

            Ok(prover)
        })
        .collect::<Result<Vec<_>, _>>()?;

    info!("synthesis time: {:?}", start.elapsed());

    // Start fft/multiexp prover timer
    let start = Instant::now();
    info!("starting proof timer");

    let input_assignments = provers
        .par_iter_mut()
        .map(|prover| {
            let input_assignment = std::mem::take(&mut prover.input_assignment);
            Arc::new(
                input_assignment
                    .into_iter()
                    .map(|s| s.to_repr())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let aux_assignments = provers
        .par_iter_mut()
        .map(|prover| {
            let aux_assignment = std::mem::take(&mut prover.aux_assignment);
            Arc::new(
                aux_assignment
                    .into_iter()
                    .map(|s| s.to_repr())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    Ok((start, provers, input_assignments, aux_assignments))
}

#[cfg(test)]
mod tests {
    use super::*;

    use blstrs::Scalar as Fr;
    use rand::Rng;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_proving_assignment_extend() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for k in &[2, 4, 8] {
            for j in &[10, 20, 50] {
                let count: usize = k * j;

                let mut full_assignment = ProvingAssignment::<Fr>::new();
                full_assignment
                    .alloc_input(|| "one", || Ok(<Fr as Field>::one()))
                    .unwrap();

                let mut partial_assignments = Vec::with_capacity(count / k);
                for i in 0..count {
                    if i % k == 0 {
                        let mut p = ProvingAssignment::new();
                        p.alloc_input(|| "one", || Ok(<Fr as Field>::one()))
                            .unwrap();
                        partial_assignments.push(p)
                    }

                    let index: usize = i / k;
                    let partial_assignment = &mut partial_assignments[index];

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el))
                            .unwrap();
                        partial_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el))
                            .unwrap();
                        partial_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    // TODO: LinearCombination
                }

                let mut combined = ProvingAssignment::new();
                combined
                    .alloc_input(|| "one", || Ok(<Fr as Field>::one()))
                    .unwrap();

                for assignment in partial_assignments.into_iter() {
                    combined.extend(assignment);
                }
                assert_eq!(combined, full_assignment);
            }
        }
    }
}


#[allow(clippy::clippy::needless_collect)]
pub fn create_proof_single_priority<E, C, P: ParameterSource<E>>(
    circuit: C,
    params: Arc<P>,
    r: E::Fr,
    s: E::Fr,
    priority: bool,
) -> Result<Proof<E>, SynthesisError>
    where
        E: MultiMillerLoop,
        C: Circuit<E::Fr> + Send,
        E::Fr: GpuName,
        E::G1Affine: GpuName,
        E::G2Affine: GpuName,
{
    info!("[create_proof_single_priority] Start create_proof!!!");

    let mut prover = ProvingAssignment::new();
    prover.alloc_input(|| "", || Ok(E::Fr::one()))?;

    start_synthesize();
    let prove_start = Instant::now();
    let part_start = Instant::now();
    info!("Before synthesize!");
    {
        circuit.synthesize(&mut prover)?;
    }

    info!("part 1 takes {:?}", part_start.elapsed());

    //start_gpu_thread();
    let part_start = Instant::now();

    for i in 0..prover.input_assignment.len() {
        prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
    }

    let worker = Worker::new();

    let vk = params.get_vk(prover.input_assignment.len())?;
    info!("prover.a.len(), to determin if it's a seal or post, {:?}", prover.a.len());


    let n = prover.a.len();
    let key = n.clone();
    let mut log_d = 0;
    while (1 << log_d) < n {
        log_d += 1;
    }

    let (post, gpu_mem_fft, gpu_mem_me_g1, gpu_mem_me_g2) = match prover.a.len()  {
        130278869 => {
            if inplace_fft(log_d) {
                (false, 4267, 2369, 4281)
            } else {
                (false, 8359, 2369, 4281)
            }
        }, //v26 32G seal
        373230 => (true, 0, 0, 0), //32G post
        4433034 => (false, 1353, 512, 512), //v26 512M seal
        320497 => (true, 0, 0, 0), //512M post
        3852162 => (false, 679, 128,128), //v26 8M seal, consumed gpu memory parameter is incorrect!
        // 512MiB
        85489 => (true, 2048, 1024, 1024),
        4423473 => (true, 2048, 1024, 1024),
        57450484 => (true, 2048, 1024, 1024),
        //8MiB
        64289 => (true, 1024, 24, 24),
        3844969 => (true, 1024, 24, 24),
        10007508 => (true, 1024, 24,24), //v28 8M seal, consumed gpu memory parameter is incorrect!
        v => {
            warn!("The bin has problem with groth16 params, can't tell if post or not by params length len:{:?}",v);
            if inplace_fft(log_d) {
                (false, 4267, 2369, 4281)
            } else {
                (false, 8359, 2369, 4281)
            }
        }
    };

    // TODO: parallelize if it's even helpful
    let input_assignment = Arc::new(
        prover
            .input_assignment
            .into_iter()
            .map(|s| s.to_repr())
            .collect::<Vec<_>>(),
    );

    let aux_assignment = Arc::new(
        prover
            .aux_assignment
            .into_iter()
            .map(|s| s.to_repr())
            .collect::<Vec<_>>(),
    );

    let a_aux_density = Arc::new(prover.a_aux_density);
    let a_aux_density_total = a_aux_density.get_total_density();

    let seg_start = Instant::now();

    let b_input_density = Arc::new(prover.b_input_density);
    let b_input_density_total = b_input_density.get_total_density();

    info!(
        "b_input_density time: {:?}",
        seg_start.elapsed(),
    );
    let seg_start = Instant::now();

    let b_aux_density = Arc::new(prover.b_aux_density);
    let b_aux_density_total = b_aux_density.get_total_density();

    info!(
        "b_aux_density time: {:?}",
        seg_start.elapsed(),
    );

    if (vk.delta_g1.is_identity() | vk.delta_g2.is_identity()).into() {
        return Err(SynthesisError::UnexpectedIdentity);
    }

    finish_syntheize();

    info!("part 2 takes {:?}", part_start.elapsed());
    let part_start = Instant::now();

    let (mut fft_kern, gpu_idx, gpued) = get_fft_kernals(log_d.clone(), post.clone(), gpu_mem_fft, priority);
    // let mut fft_kern = LockedFftKernel::new(priority);
    // let gpu_idx = 0;
    // let gpued = false;
    info!("[kernal-release], start fft!!!");
    #[cfg(any(feature = "cuda", feature = "opencl"))]
        let prio_lock = if priority {
        trace!("acquiring priority lock");
        Some(PriorityLock::lock())
    } else {
        None
    };

    let a = {
        let mut a = EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new()))?;
        let mut b = EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new()))?;
        let mut c = EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new()))?;

        EvaluationDomain::ifft_many(&mut [&mut a, &mut b, &mut c], &worker, &mut fft_kern)?;
        EvaluationDomain::coset_fft_many(&mut [&mut a, &mut b, &mut c], &worker, &mut fft_kern)?;

        a.mul_assign(&worker, &b);
        drop(b);
        a.sub_assign(&worker, &c);
        drop(c);

        a.divide_by_z_on_coset(&worker);
        a.icoset_fft(&worker, &mut fft_kern)?;

        drop(fft_kern);
        if gpued {
            info!("[kernal-release]end, release fft GPU memory!");
            finish_use_gpu(gpu_idx, gpu_mem_fft);
        } else if GPU_BELL() {
            info!("Release fft thread for a");
            cpu_finish();
        }

        let a = a.into_coeffs();
        let a_len = a.len() - 1;
        let a = a
            .into_par_iter()
            .take(a_len)
            .map(|s| s.to_repr())
            .collect::<Vec<_>>();
        Arc::new(a)
    };


    info!("part 3 takes {:?}", part_start.elapsed());

    let part_start = Instant::now();

    let (h, l, a_inputs, a_aux, b_g1_inputs, b_g1_aux, b_g2_inputs, b_g2_aux) = if (GPU_BELL()) && (PAR_BELL()) && !post {
        info!("In the parellel mutliple exponent mode");
        crossbeam::thread::scope(|s| -> Result<_, ()> {
            let params_h = params.get_h_cached(a.len(), key).unwrap();
            let h_handler = s.spawn(move |_|  {
                info!("Start compute G1 H");
                let (mut multiexp_kern, gpu_idx, gpued) = get_me_kernals::<E>(n, post, gpu_mem_me_g1, priority);
                let mut multiexp_kern = multiexp_kern.unwrap();
                info!("[kernal-release]me-k, start G1 H!!!");
                let start_h = Instant::now();
                let worker = Worker::new();
                let h = multiexp(
                    &worker,
                    params_h,
                    FullDensity,
                    a,
                    &mut multiexp_kern,
                );
                info!("Compute H takes: {:?}", start_h.elapsed());
                drop(multiexp_kern);
                if gpued {
                    info!("[kernal-release]end, release G1 H GPU memory!");
                    finish_use_gpu(gpu_idx, gpu_mem_me_g1);
                }
                else if GPU_BELL() {
                    info!("Release CPU thread for h");
                    cpu_finish();
                }

                h
            });

            let aux_assignment_l = aux_assignment.clone();
            let params_l = params.get_l_cached(aux_assignment.len(), key).unwrap();
            let l_handler = s.spawn(move |_|  {
                info!("Start compute G1 L");
                //let op_cpu = !cpu_running();
                // let (mut multiexp_kern, gpu_idx, gpued) = LockedMultiexpKernel::<E::G1Affine>::new(priority);
                // let mut multiexp_kern = LockedMultiexpKernel::<E::G1Affine>::new(priority);
                let (mut multiexp_kern, gpu_idx, gpued) = get_me_kernals::<E>(n, post, gpu_mem_me_g1, priority);
                let mut multiexp_kern = multiexp_kern.unwrap();
                info!("[kernal-release]me-k, start!");
                let start_l = Instant::now();
                let worker = Worker::new();
                let l = multiexp(
                    &worker,
                    params_l,
                    FullDensity,
                    aux_assignment_l,
                    &mut multiexp_kern,
                );
                info!("Compute L takes: {:?}", start_l.elapsed());
                drop(multiexp_kern);

                if gpued {
                    info!("[kernal-release]end, release l GPU memory!");
                    finish_use_gpu(gpu_idx, gpu_mem_me_g1);
                }
                else if GPU_BELL() {
                    info!("Release CPU thread for l");
                    cpu_finish();
                }
                l
            });


            let b_aux_density_oth = b_aux_density.clone();
            let b_input_density_oth = b_input_density.clone();
            let input_assignment_oth = input_assignment.clone();
            let aux_assignment_oth = aux_assignment.clone();

            let (a_inputs_source, a_aux_source) =
                params.get_a_cached(input_assignment.len(), a_aux_density_total, key).unwrap();
            let (b_g1_inputs_source, b_g1_aux_source) =
                params.get_b_g1_cached(b_input_density_total, b_aux_density_total, key).unwrap();
            let other_g1_handler = s.spawn(move |_|  {
                info!("Start compute G1 others");
                //let op_cpu = !cpu_running();
                // let (mut multiexp_kern, gpu_idx, gpued) = LockedMultiexpKernel::<E::G1Affine>::new(priority);
                // let mut multiexp_kern = LockedMultiexpKernel::<E::G1Affine>::new(priority);
                // let gpu_idx = 0;
                // let gpued = false;
                let (mut multiexp_kern, gpu_idx, gpued) = get_me_kernals::<E>(n, post, gpu_mem_me_g1, priority);
                let mut multiexp_kern = multiexp_kern.unwrap();
                info!("[kernal-release]me-k, start G1!");
                let start_1 = Instant::now();
                let worker = Worker::new();
                let a_inputs = multiexp(
                    &worker,
                    a_inputs_source,
                    FullDensity,
                    input_assignment_oth.clone(),
                    &mut multiexp_kern,
                );

                info!("multiexp a_aux");

                let a_aux = multiexp(
                    &worker,
                    a_aux_source,
                    a_aux_density,
                    aux_assignment_oth.clone(),
                    &mut multiexp_kern,
                );


                info!("Start multiexp b_g1_inputs");

                let b_g1_inputs = multiexp(
                    &worker,
                    b_g1_inputs_source,
                    b_input_density_oth,
                    input_assignment_oth,
                    &mut multiexp_kern,
                );

                info!("Start multiexp b_g1_aux");
                let b_g1_aux = multiexp(
                    &worker,
                    b_g1_aux_source,
                    b_aux_density_oth,
                    aux_assignment_oth,
                    &mut multiexp_kern,
                );

                info!("Compute other G1 takes: {:?}", start_1.elapsed());
                drop(multiexp_kern);
                if gpued {
                    info!("[kernal-release]end, release G1 GPU memory!");
                    finish_use_gpu(gpu_idx, gpu_mem_me_g1);
                } else if GPU_BELL() {
                    info!("Release CPU thread for g1");
                    cpu_finish();
                }

                (a_inputs, a_aux, b_g1_inputs, b_g1_aux)
            });

            let (b_g2_inputs_source, b_g2_aux_source) =
                params.get_b_g2_cached(b_input_density_total, b_aux_density_total, key).unwrap();
            let g2_handler = s.spawn(move |_|  {
                info!("Start compute G2");
                //let op_cpu = !cpu_running();
                // let (mut multiexp_kern, gpu_idx, gpued) = LockedMultexpKernel::<E::G2Affine>::new(priority);
                // let mut multiexp_kern = LockedMultiexpKernel::<E::G2Affine>::new(priority);
                let (mut multiexp_kern, gpu_idx, gpued) = get_me_kernals_g2::<E>(n, post, gpu_mem_me_g1, priority);
                let mut multiexp_kern = multiexp_kern.unwrap();
                info!("[kernal-release]me-k, start G2!");
                let g2_start = Instant::now();
                info!("multiexp b_g2_inputs");
                let worker = Worker::new();
                let b_g2_inputs = multiexp(
                    &worker,
                    b_g2_inputs_source,
                    b_input_density,
                    input_assignment,
                    &mut multiexp_kern
                );
                info!("multiexp b_g2_aux");

                let b_g2_aux = multiexp(&worker, b_g2_aux_source, b_aux_density.clone(), aux_assignment.clone(), &mut multiexp_kern);
                info!("Multiexp g2 time: {:?}", g2_start.elapsed());
                drop(multiexp_kern);
                if gpued {
                    info!("[kernal-release]end, release G2 GPU memory!");
                    finish_use_gpu(gpu_idx, gpu_mem_me_g2);
                }
                else if GPU_BELL() {
                    info!("Release CPU thread for g2");
                    cpu_finish();
                }
                (b_g2_inputs, b_g2_aux)
            });

            let h = h_handler.join().unwrap();
            let l = l_handler.join().unwrap();
            let (a_inputs, a_aux, b_g1_inputs, b_g1_aux) = other_g1_handler.join().unwrap();
            let (b_g2_inputs, b_g2_aux) = g2_handler.join().unwrap();

            Ok((h, l, a_inputs, a_aux, b_g1_inputs, b_g1_aux, b_g2_inputs, b_g2_aux))
        }).unwrap().unwrap()

    } else {
        let (mut multiexp_kern, gpu_idx, gpued) = get_me_kernals_g2::<E>(n, post, gpu_mem_me_g1, priority);
        let mut multiexp_kern = multiexp_kern.unwrap();
        // let gpu_idx = 0;
        // let gpued = false;
        // let mut multiexp_kern = LockedMultiexpKernel::<E::G2Affine>::new(priority);
        info!("[kernal-release]me-k start, start G2!");
        info!("gpu_idx 2 : {}, post: {}", gpu_idx, post);
        let g2_start = Instant::now();
        let param_start = Instant::now();
        let (b_g2_inputs_source, b_g2_aux_source) =
            params.get_b_g2_cached(b_input_density_total, b_aux_density_total, key)?;
        info!("param get b g2 takes: {:?}", param_start.elapsed());

        info!("multiexp b_g2_inputs");
        let b_g2_inputs = multiexp(
            &worker,
            b_g2_inputs_source,
            b_input_density.clone(),
            input_assignment.clone(),
            &mut multiexp_kern
        );
        info!("multiexp b_g2_aux");

        let b_g2_aux = multiexp(
            &worker,
            b_g2_aux_source,
            b_aux_density.clone(),
            aux_assignment.clone(),
            &mut multiexp_kern);

        info!("Multiexp g2 time: {:?}", g2_start.elapsed());

        // _multiexp_kern_tmp = None;
        drop(multiexp_kern);

        if (GPU_BELL()) && gpued {
            info!("[kernal-release]mek-k end, release G2 GPU memory!");
            finish_use_gpu(gpu_idx, gpu_mem_me_g2);
        } else if GPU_BELL() {
            info!("Release CPU thread for b_g2_aux");
            cpu_finish();
        }

        // let (mut multiexp_kern, gpu_idx, gpued) = LockedMultiexpKernel::<E::G1Affine>::new(priority);
        let (mut multiexp_kern, gpu_idx, gpued) = get_me_kernals::<E>(n, post, gpu_mem_me_g1, priority);
        let mut multiexp_kern = multiexp_kern.unwrap();
        // let mut multiexp_kern = LockedMultiexpKernel::<E::G1Affine>::new(priority);
        // let mut multiexp_kern = multiexp_kern.unwrap();
        info!("[kernal-release]me-k, start H!");
        info!("Start multiexp h");
        let g1_start = Instant::now();
        let start_h = Instant::now();
        let params_h = params.get_h_cached(a.len(), key).unwrap();
        let h = multiexp(
            &worker,
            params_h,
            FullDensity,
            a,
            &mut multiexp_kern,
        );
        info!("Compute H takes: {:?}", start_h.elapsed());
        info!("Start multiexp l");

        let start_l = Instant::now();
        let params_l = params.get_l_cached(aux_assignment.len(), key).unwrap();
        let l = multiexp(
            &worker,
            params_l,
            FullDensity,
            aux_assignment.clone(),
            &mut multiexp_kern,
        );
        info!("Compute L takes: {:?}", start_l.elapsed());
        info!("Start multiexp a_inputs");

        let param_start = Instant::now();
        let (a_inputs_source, a_aux_source) =
            params.get_a_cached(input_assignment.len(), a_aux_density_total, key)?;
        info!("param get a takes: {:?}", param_start.elapsed());
        if CACHE_BUILDING.load(Ordering::SeqCst) == 1 {
            CACHE_BUILDING.fetch_sub(1, Ordering::SeqCst);
        }

        let a_inputs = multiexp(
            &worker,
            a_inputs_source,
            FullDensity,
            input_assignment.clone(),
            &mut multiexp_kern,
        );

        info!("multiexp a_aux");

        let a_aux = multiexp(
            &worker,
            a_aux_source,
            a_aux_density,
            aux_assignment.clone(),
            &mut multiexp_kern,
        );

        info!("Start multiexp b_g1_inputs");
        let param_start = Instant::now();
        let (b_g1_inputs_source, b_g1_aux_source) =
            params.get_b_g1_cached(b_input_density_total, b_aux_density_total, key)?;
        info!("param get b g1 takes: {:?}", param_start.elapsed());
        let b_g1_inputs = multiexp(
            &worker,
            b_g1_inputs_source,
            b_input_density.clone(),
            input_assignment.clone(),
            &mut multiexp_kern,
        );
        info!("Start multiexp b_g1_aux");

        let b_g1_aux = multiexp(
            &worker,
            b_g1_aux_source,
            b_aux_density.clone(),
            aux_assignment.clone(),
            &mut multiexp_kern,
        );
        info!("Multiexp g1 time: {:?}", g1_start.elapsed());

        if (vk.delta_g1.is_identity() | vk.delta_g2.is_identity()).into() {
            return Err(SynthesisError::UnexpectedIdentity);
        }

        drop(multiexp_kern);

        if GPU_BELL() && gpued {
            info!("[kernal-release]mek-k end, release H GPU memory!");
            finish_use_gpu(gpu_idx, gpu_mem_me_g1);
        } else if GPU_BELL() {
            info!("Release CPU thread for b_g1_aux");
            cpu_finish();
        }
        info!("finished multiexp!");
        (h, l, a_inputs, a_aux, b_g1_inputs, b_g1_aux, b_g2_inputs, b_g2_aux)
    };

    //finish_gpu_thread();

    #[cfg(any(feature = "cuda", feature = "opencl"))]
        {
            trace!("dropping priority lock");
            drop(prio_lock);
        }
    info!("part 4 takes {:?}", part_start.elapsed());
    let part_start = Instant::now();

    let mut g_a = vk.delta_g1.mul(r);
    g_a.add_assign(&vk.alpha_g1);
    let mut g_b = vk.delta_g2.mul(s);
    g_b.add_assign(&vk.beta_g2);
    let mut g_c;
    {
        let mut rs = r;
        rs.mul_assign(&s);

        g_c = vk.delta_g1.mul(rs);
        g_c.add_assign(&vk.alpha_g1.mul(s));
        g_c.add_assign(&vk.beta_g1.mul(r));
    }
    let mut a_answer = a_inputs.wait()?;
    a_answer.add_assign(&a_aux.wait()?);
    g_a.add_assign(&a_answer);
    a_answer.mul_assign(s);
    g_c.add_assign(&a_answer);

    let mut b1_answer = b_g1_inputs.wait()?;
    b1_answer.add_assign(&b_g1_aux.wait()?);
    let mut b2_answer = b_g2_inputs.wait()?;
    b2_answer.add_assign(&b_g2_aux.wait()?);

    g_b.add_assign(&b2_answer);
    b1_answer.mul_assign(r);
    g_c.add_assign(&b1_answer);
    g_c.add_assign(&h.wait()?);
    g_c.add_assign(&l.wait()?);
    info!("part 5 takes {:?}", part_start.elapsed());
    info!("zksnark takes {:?}", prove_start.elapsed());
    Ok(Proof {
        a: g_a.to_affine(),
        b: g_b.to_affine(),
        c: g_c.to_affine(),
    })

}

pub fn inplace_fft(fft_n: usize) -> bool {
    info!("fft_n: {:?}", fft_n);
    if fft_n == FFT_N_VALUE {
        true
    } else {
        false
    }
}

pub fn get_me_kernals<E: Engine>(_n: usize, post:bool, mem: usize, priority: bool) -> (Option<LockedMultiexpKernel<'static, E::G1Affine>>, usize, bool)
    where 
        <E as Engine>::G1Affine: GpuName
{
    let start_time = Instant::now();
    if post {
        info!("post me kernal!");
        let post_kernal = Some(LockedMultiexpKernel::<E::G1Affine>::new(priority, None));
        return (post_kernal, 99, false)
    } else {
        info!("[get_me_kernals] get!!!!!");
        if GPU_BELL() {
            bell_start();
            let mut rng = rand::thread_rng();
            loop {
                if !hash_rnning() {
                    let rand_sleep = rng.gen_range(0..10);
                    debug!("Random sleep {:?} MS", rand_sleep);
                    sleep(Duration::from_millis(rand_sleep));
                    let (gpu_ok, min_idx, gpu_id) = free_gpu_ok(mem);
                    if gpu_ok {
                        if !gpu_overloaded(min_idx) {
                            let kern_2 = {
                                Some(LockedMultiexpKernel::<E::G1Affine>::new(priority, Some(min_idx)))
                            };
                            bell_finish();
                            info!("GPU get_me_kernals takes: {:?}", start_time.elapsed());
                            return (kern_2, min_idx, true)
                        } else {
                            let start_sub = Instant::now();
                            finish_use_gpu(min_idx, mem);
                            info!("BELL fetch sub takes: {:?}, abnormal situation", start_sub.elapsed());
                            let sleep_time = Duration::from_millis(WAIT_GPU);
                            debug!("Competition happend, all GPU is full for Multiexp, sleep {:?} MS", WAIT_GPU);
                            sleep(sleep_time);
                        }
                    } else {
                        if should_opt_cpu() {
                            debug!("all GPU is full for Multiexp, opt_cpu told us to run on cpu don't sleep but use cpu");
                            cpu_start();
                            bell_finish();
                            return (None, 99, false)
                        } else {
                            let sleep_time = Duration::from_millis(WAIT_GPU);
                            debug!("all GPU is full for Multiexp, sleep {:?} MS", WAIT_GPU);
                            sleep(sleep_time);
                        }
                    }
                } else {
                    if should_opt_cpu() {
                        debug!("all GPU is full for Multiexp, opt_cpu told us to run on cpu don't sleep but use cpu");
                        cpu_start();
                        bell_finish();
                        return (None, 99, false)
                    } else {
                        let sleep_time = Duration::from_millis(WAIT_GPU);
                        debug!("hashing thread is using GPU, sleep {:?} MS", WAIT_GPU);
                        sleep(sleep_time);
                    }
                }
            }
        }
        else {
            info!("Use CPU");
            return ( None, 99, false)
        };
    }

}

pub fn get_me_kernals_g2<E: Engine>(_n: usize, post:bool, mem: usize, priority: bool) -> (Option<LockedMultiexpKernel<'static, E::G2Affine>>, usize, bool)
    where 
        <E as Engine>::G2Affine: GpuName
{
    let start_time = Instant::now();
    if post {
        info!("post me kernal!");
        let post_kernal = Some(LockedMultiexpKernel::<E::G2Affine>::new(priority, None));
        return (post_kernal, 99, false)
    } else {
        info!("[get_me_kernals] get!!!!!");
        if GPU_BELL() {
            bell_start();
            let mut rng = rand::thread_rng();
            loop {
                if !hash_rnning() {
                    let rand_sleep = rng.gen_range(0..10);
                    debug!("Random sleep {:?} MS", rand_sleep);
                    sleep(Duration::from_millis(rand_sleep));
                    let (gpu_ok, min_idx, gpu_id) = free_gpu_ok(mem);
                    if gpu_ok {
                        if !gpu_overloaded(min_idx) {
                            let kern_2 = {
                                Some(LockedMultiexpKernel::<E::G2Affine>::new(priority, Some(min_idx)))
                            };
                            bell_finish();
                            info!("GPU get_me_kernals takes: {:?}", start_time.elapsed());
                            return (kern_2, min_idx, true)
                        } else {
                            let start_sub = Instant::now();
                            finish_use_gpu(min_idx, mem);
                            info!("BELL fetch sub takes: {:?}, abnormal situation", start_sub.elapsed());
                            let sleep_time = Duration::from_millis(WAIT_GPU);
                            debug!("Competition happend, all GPU is full for Multiexp, sleep {:?} MS", WAIT_GPU);
                            sleep(sleep_time);
                        }
                    } else {
                        if should_opt_cpu() {
                            debug!("all GPU is full for Multiexp, opt_cpu told us to run on cpu don't sleep but use cpu");
                            cpu_start();
                            bell_finish();
                            return (None, 99, false)
                        } else {
                            let sleep_time = Duration::from_millis(WAIT_GPU);
                            debug!("all GPU is full for Multiexp, sleep {:?} MS", WAIT_GPU);
                            sleep(sleep_time);
                        }
                    }
                } else {
                    if should_opt_cpu() {
                        debug!("all GPU is full for Multiexp, opt_cpu told us to run on cpu don't sleep but use cpu");
                        cpu_start();
                        bell_finish();
                        return (None, 99, false)
                    } else {
                        let sleep_time = Duration::from_millis(WAIT_GPU);
                        debug!("hashing thread is using GPU, sleep {:?} MS", WAIT_GPU);
                        sleep(sleep_time);
                    }
                }
            }
        }
        else {
            info!("Use CPU");
            return ( None, 99, false)
        };
    }

}

pub fn finish_syntheize() {
    let thread_num = RNNING_SYN_THREAD_NUM.fetch_sub(1, Ordering::SeqCst);
    info!("finish_syntheize_thread, current running synthesize threads is {:?}", thread_num);
}

pub fn start_gpu_thread() {
    let thread_num = RNNING_GPU_THREAD_NUM.fetch_add(1, Ordering::SeqCst);
    info!("start_gpu_thread, current running gpu threads is {:?}", thread_num);
}

pub fn finish_gpu_thread() {
    let thread_num = RNNING_GPU_THREAD_NUM.fetch_sub(1, Ordering::SeqCst);
    info!("finish_gpu_thread, current running gpu threads is {:?}", thread_num);
}

pub fn sleep_time(sleep_number: u32) -> u64{
    match sleep_number {
        0..=10 => 6000,
        11..=50 => 3000,
        51..=100 => 1500,
        101..=500 => 800,
        501..=1000 => 400,
        1001..=2001 => 200,
        _ => 100,

    }
}

pub fn start_synthesize() {
    let mut sleep_number = 0;
    let this_thread_num = THIS_THREAD_NUM.fetch_add(1, Ordering::SeqCst);
    loop {
        let mut rng = rand::thread_rng();

        let sleep_time = rng.gen_range(0..SYNTHESIZE_SLEEP_TIME());
        debug!("sleep {:?} MS to wait synthesize for thread num {:?}!", sleep_time, this_thread_num);
        if (RNNING_SYN_THREAD_NUM.load(Ordering::SeqCst) < NUM_PROVING_THREAD()) &&
            ( (get_waiting_bell() <= MAX_BELL_GPU_THREAD_NUM()) || (RNNING_SYN_THREAD_NUM.load(Ordering::SeqCst) == 0 && MAX_BELL_GPU_THREAD_NUM()>=4))
            && (this_thread_num == RNNING_THREAD_NUM.load(Ordering::SeqCst)+1)
            && (CACHE_BUILDING.load(Ordering::SeqCst) == 0){
            RNNING_THREAD_NUM.fetch_add(1, Ordering::SeqCst);
            RNNING_SYN_THREAD_NUM.fetch_add(1, Ordering::SeqCst);
            if this_thread_num == 1 {
                CACHE_BUILDING.fetch_add(1, Ordering::SeqCst);
            }
            info!("MAX_BELL_GPU_THREAD_NUM is {:?}", MAX_BELL_GPU_THREAD_NUM());
            info!("Thread num {:?} get through to start synthesize with running synthesize thread {:?} and waiting bellman tasks {:?}", this_thread_num, RNNING_SYN_THREAD_NUM.load(Ordering::SeqCst), get_waiting_bell());
            return
        } else {
            sleep(Duration::from_millis(sleep_time as u64));
        }
    }
}

pub fn get_fft_kernals<F: ec_gpu::GpuName + ff::Field>(log_d: usize, post: bool, mem: usize, priority: bool) -> (Option<LockedFftKernel<'static, F>>, usize, bool)
   
{
    if post {
        (Some(LockedFftKernel::new(priority, None)), 99, false)
    } else {
        info!("[get_fft_kernals] get!!!!");
        if GPU_BELL() {
            bell_start();
            let mut rng = rand::thread_rng();
            loop {
                if !hash_rnning() {
                    let rand_sleep = rng.gen_range(0..10);
                    debug!("Random sleep {:?} MS", rand_sleep);
                    sleep(Duration::from_millis(rand_sleep));
                    let (gpu_ok, min_idx, gpu_id) = free_gpu_ok(mem);
                    if gpu_ok {
                        if !gpu_overloaded(min_idx) {
                            let kern_1 = if inplace_fft(log_d) {
                                let res = {
                                    LockedFftKernel::new(priority, Some(min_idx))
                                };
                                res
                            } else {
                                LockedFftKernel::new(priority, Some(min_idx))
                            };
                            bell_finish();
                            return (Some(kern_1), min_idx, true)
                        } else {
                            let start_sub = Instant::now();
                            finish_use_gpu(min_idx, mem);
                            info!("BELL fetch sub takes: {:?}, abnormal situation", start_sub.elapsed());
                            let sleep_time = Duration::from_millis(WAIT_GPU);
                            debug!("Competition happend, all GPU is full for FFT, sleep {:?} MS", WAIT_GPU);
                            sleep(sleep_time);
                        }
                    } else {
                        if should_opt_cpu() {
                            debug!("all GPU is full for FFT, opt_cpu told us to run on cpu don't sleep but use cpu");
                            cpu_start();
                            bell_finish();
                            return ( None, 99, false)
                        } else {
                            let sleep_time = Duration::from_millis(WAIT_GPU);
                            debug!("all GPU is full for FFT, sleep {:?} MS", WAIT_GPU);
                            sleep(sleep_time);
                        }
                    }
                } else {
                    if should_opt_cpu() {
                        debug!("all GPU is full for FFT, opt_cpu told us to run on cpu don't sleep but use cpu");
                        cpu_start();
                        bell_finish();
                        return (None, 99, false)
                    }
                    else {
                        let sleep_time = Duration::from_millis(WAIT_GPU);
                        info!("hashing thread is using GPU, sleep {:?} MS", WAIT_GPU);
                        sleep(sleep_time);
                    }
                }
            }
        }
        else {
            info!("Use CPU");
            return (None, 99, false)
        };
    }

}

#[allow(clippy::needless_collect)]
pub fn create_random_proof_batch_priority_with_verify<E, C, R, P: ParameterSource<E>>(
    circuits: Vec<C>,
    circuits_retry: Vec<C>,
    params: P,
    rng: &mut R,
    priority: bool,
    public_inputs: &[Vec<E::Fr>],
    pvk: &PreparedVerifyingKey<E>,
) -> Result<Vec<Proof<E>>, SynthesisError>
    where
        E: MultiMillerLoop,
        C: Circuit<E::Fr> + Send,
        E::Fr: GpuName,
        E::G1Affine: GpuName,
        E::G2Affine: GpuName,
        R: RngCore,
{
    info!("Start a batch zksnark creation");
    if NO_CUSTOM() || priority {
        info!("Do NOT use cusomization zksnark routine!!");
        let r_s = (0..circuits.len())
            .map(|_| E::Fr::random(&mut *rng))
            .collect();
        let s_s = (0..circuits.len())
            .map(|_| E::Fr::random(&mut *rng))
            .collect();

        create_proof_batch_priority::<E, C, P>(circuits, params, r_s, s_s, priority)
    } else {
        info!("Do use cusomization zksnark routine!!");

        let r = E::Fr::random(&mut *rng);
        let s = E::Fr::random(&mut *rng);

        info!("r value is {:?}", r);
        info!("s value is {:?}", s);

        info!("circuits length is {:?}", circuits.len());

        let arc_params = Arc::new(params);
        let res = crossbeam::thread::scope(|scope| -> Vec<Proof<E>> {
            let handlers: Vec<_> = circuits.into_iter().zip(public_inputs.into_iter()).zip(circuits_retry.into_iter()).map(|((circuit, public_input), circuit_retry)| {
                let r_cloned = r.clone();
                let s_cloned = s.clone();
                let p_cloned = priority.clone();
                let arc_params_clone = arc_params.clone();
                let r_retry = E::Fr::random(&mut *rng);
                let s_retry = E::Fr::random(&mut *rng);

                scope.spawn(move |_| {

                    let res = create_proof_single_priority(circuit, arc_params_clone.clone(), r_cloned, s_cloned, p_cloned).unwrap();
                    let verify_res = verify_proof(pvk, &res, public_input).unwrap();
                    if verify_res == false {
                        info!("Retry this particular zksnark coz it's failed!!!!!!!!!");
                        let res_retry = create_proof_single_priority(circuit_retry, arc_params_clone, r_retry, s_retry, p_cloned).unwrap();
                        res_retry
                    } else {
                        info!("Bellman verify partition success");
                        res
                    }
                })
            }).collect();
            handlers.into_iter().map(|handler| {
                handler.join().unwrap()
            }).collect()
        }).unwrap();
        info!("r value is {:?}", r);
        info!("s value is {:?}", s);
        info!("Before end!!!");
        Ok(res)
    }
}