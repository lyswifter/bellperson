use group::{prime::PrimeCurveAffine, UncompressedEncoding};
use pairing::MultiMillerLoop;

use crate::SynthesisError;

#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
use rayon::prelude::*;

use std::fs::File;
use std::io;
use std::mem;
use blstrs::{Bls12};
use pairing::Engine;
use log::info;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use super::{ParameterSource, PreparedVerifyingKey, VerifyingKey};


lazy_static::lazy_static! {
    pub static ref PARAMS_H_CACHE: Arc<Mutex<HashMap<usize, (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize)>>> = Arc::new(Mutex::new(HashMap::new()));
    pub static ref PARAMS_L_CACHE: Arc<Mutex<HashMap<usize, (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize)>>> = Arc::new(Mutex::new(HashMap::new()));

    pub static ref PARAMS_A_CACHE: Arc<Mutex<HashMap<usize, ((Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize))>>> = Arc::new(Mutex::new(HashMap::new()));
    pub static ref PARAMS_B_G1_CACHE: Arc<Mutex<HashMap<usize, ((Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize))>>> = Arc::new(Mutex::new(HashMap::new()));
    pub static ref PARAMS_B_G2_CACHE: Arc<Mutex<HashMap<usize, ((Arc<Vec<<Bls12 as Engine>::G2Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G2Affine>>, usize))>>> = Arc::new(Mutex::new(HashMap::new()));
}

pub struct MappedParameters<E>
where
    E: MultiMillerLoop,
{
    /// The parameter file we're reading from.  
    pub param_file_path: PathBuf,
    /// The file descriptor we have mmaped.
    pub param_file: File,
    /// The actual mmap.
    pub params: Mmap,

    /// This is always loaded (i.e. not lazily loaded).
    pub vk: VerifyingKey<E>,
    pub pvk: PreparedVerifyingKey<E>,

    /// Elements of the form ((tau^i * t(tau)) / delta) for i between 0 and
    /// m-2 inclusive. Never contains points at infinity.
    pub h: Vec<Range<usize>>,

    /// Elements of the form (beta * u_i(tau) + alpha v_i(tau) + w_i(tau)) / delta
    /// for all auxiliary inputs. Variables can never be unconstrained, so this
    /// never contains points at infinity.
    pub l: Vec<Range<usize>>,

    /// QAP "A" polynomials evaluated at tau in the Lagrange basis. Never contains
    /// points at infinity: polynomials that evaluate to zero are omitted from
    /// the CRS and the prover can deterministically skip their evaluation.
    pub a: Vec<Range<usize>>,

    /// QAP "B" polynomials evaluated at tau in the Lagrange basis. Needed in
    /// G1 and G2 for C/B queries, respectively. Never contains points at
    /// infinity for the same reason as the "A" polynomials.
    pub b_g1: Vec<Range<usize>>,
    pub b_g2: Vec<Range<usize>>,

    pub checked: bool,
}

impl<'a, E> ParameterSource<E> for &'a MappedParameters<E>
where
    E: MultiMillerLoop,
{
    type G1Builder = (Arc<Vec<E::G1Affine>>, usize);
    type G2Builder = (Arc<Vec<E::G2Affine>>, usize);

    fn get_vk(&self, _: usize) -> Result<&VerifyingKey<E>, SynthesisError> {
        Ok(&self.vk)
    }

    fn get_h(&self, _num_h: usize) -> Result<Self::G1Builder, SynthesisError> {
        let builder = self
            .h
            .par_iter()
            .cloned()
            .map(|h| read_g1::<E>(&self.params, h, self.checked))
            .collect::<Result<_, _>>()?;

        Ok((Arc::new(builder), 0))
    }

    fn get_l(&self, _num_l: usize) -> Result<Self::G1Builder, SynthesisError> {
        let builder = self
            .l
            .par_iter()
            .cloned()
            .map(|l| read_g1::<E>(&self.params, l, self.checked))
            .collect::<Result<_, _>>()?;

        Ok((Arc::new(builder), 0))
    }

    fn get_a(
        &self,
        num_inputs: usize,
        _num_a: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        let builder = self
            .a
            .par_iter()
            .cloned()
            .map(|a| read_g1::<E>(&self.params, a, self.checked))
            .collect::<Result<_, _>>()?;

        let builder: Arc<Vec<_>> = Arc::new(builder);

        Ok(((builder.clone(), 0), (builder, num_inputs)))
    }

    fn get_b_g1(
        &self,
        num_inputs: usize,
        _num_b_g1: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        let builder = self
            .b_g1
            .par_iter()
            .cloned()
            .map(|b_g1| read_g1::<E>(&self.params, b_g1, self.checked))
            .collect::<Result<_, _>>()?;

        let builder: Arc<Vec<_>> = Arc::new(builder);

        Ok(((builder.clone(), 0), (builder, num_inputs)))
    }

    fn get_b_g2(
        &self,
        num_inputs: usize,
        _num_b_g2: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        let builder = self
            .b_g2
            .par_iter()
            .cloned()
            .map(|b_g2| read_g2::<E>(&self.params, b_g2, self.checked))
            .collect::<Result<_, _>>()?;

        let builder: Arc<Vec<_>> = Arc::new(builder);

        Ok(((builder.clone(), 0), (builder, num_inputs)))
    }
    fn get_h_cached(&self, _num_h: usize, key: usize) -> Result<Self::G1Builder, SynthesisError>
    {
        info!("_num_h: {:?}", _num_h);
        info!("cache key: {:?}", key);
        let mut params_h_cache = (*PARAMS_H_CACHE).lock().unwrap();
        let cached_h = params_h_cache.get(&key);
        unsafe {
            match cached_h {
                Some(params) => {
                    info!("There is already a cached entry in cache h");
                    let h_raw = params.clone();
                    unsafe {
                        let h = mem::transmute::<(Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize), Self::G1Builder>(h_raw);
                        Ok(h)
                    }
                },
                None => {
                    info!("nothing cached, build new cache h");
                    let h = self.get_h(_num_h).unwrap();
                    unsafe {
                        let h_raw = mem::transmute::<Self::G1Builder, (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize)>(h.clone());
                        params_h_cache.insert(key, h_raw);
                    }
                    Ok(h)
                }
            }
        }
    }
    fn get_l_cached(&self, _num_l: usize, key: usize) -> Result<Self::G1Builder, SynthesisError> {
        info!("_num_l: {:?}", _num_l);
        info!("cache key: {:?}", key);
        let mut params_l_cache = (*PARAMS_L_CACHE).lock().unwrap();
        let cached_l = params_l_cache.get(&key);
        unsafe {
            match cached_l {
                Some(params) => {
                    info!("There is already a cached entry in cache l");
                    let l_raw = params.clone();
                    unsafe {
                        let l = mem::transmute::<(Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize), Self::G1Builder>(l_raw);
                        Ok(l)
                    }
                },
                None => {
                    info!("nothing cached, build new cache l");
                    let l = self.get_l(_num_l).unwrap();
                    unsafe {
                        let l_raw = mem::transmute::<Self::G1Builder, (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize)>(l.clone());
                        params_l_cache.insert(key, l_raw);
                    }
                    Ok(l)
                }
            }
        }
    }
    fn get_a_cached(
        &self,
        num_inputs: usize,
        _num_a: usize,
        key: usize
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        info!("Before acquire lock for cache a");
        info!("cache key: {:?}", key);
        let mut params_a_cache = (*PARAMS_A_CACHE).lock().unwrap();
        let cached_a = params_a_cache.get(&key);
        unsafe {
            match cached_a {
                Some(params) => {
                    info!("There is already a cached entry in cache a");
                    let a_raw = params.clone();
                    unsafe {
                        let a = mem::transmute::<((Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize)),
                            (Self::G1Builder, Self::G1Builder)>(a_raw);
                        Ok(a)
                    }
                },
                None => {
                    info!("nothing cached, build new cache a");
                    let a = self.get_a(num_inputs, _num_a).unwrap();
                    unsafe {
                        let a_raw = mem::transmute::<(Self::G1Builder, Self::G1Builder),
                            ((Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize))>(a.clone());
                        params_a_cache.insert(key, a_raw);
                    }
                    Ok(a)
                }
            }
        }
    }
    fn get_b_g1_cached(
        &self,
        num_inputs: usize,
        _num_b_g1: usize,
        key: usize
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        info!("Before acquire lock for cache b g1");
        info!("cache key: {:?}", key);
        let mut params_b_g1_cache = (*PARAMS_B_G1_CACHE).lock().unwrap();
        let cached_b_g1 = params_b_g1_cache.get(&key);
        unsafe {
            match cached_b_g1 {
                Some(params) => {
                    info!("There is already a cached entry in cache b_g1");
                    let b_g1_raw = params.clone();
                    unsafe {
                        let b_g1 = mem::transmute::<((Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize)),
                            (Self::G1Builder, Self::G1Builder)>(b_g1_raw);
                        Ok(b_g1)
                    }
                },
                None => {
                    info!("nothing cached, build new cache b_g1");
                    let b_g1 = self.get_b_g1(num_inputs, _num_b_g1).unwrap();
                    unsafe {
                        let b_g1_raw = mem::transmute::<(Self::G1Builder, Self::G1Builder),
                            ((Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G1Affine>>, usize))>(b_g1.clone());
                        params_b_g1_cache.insert(key, b_g1_raw);
                    }
                    Ok(b_g1)
                }
            }
        }
    }
    fn get_b_g2_cached(
        &self,
        num_inputs: usize,
        _num_b_g2: usize,
        key: usize
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        info!("Before acquire lock for cache b g2");
        info!("cache key: {:?}", key);
        let mut params_b_g2_cache = (*PARAMS_B_G2_CACHE).lock().unwrap();
        let cached_b_g2 = params_b_g2_cache.get(&key);
        unsafe {
            match cached_b_g2 {
                Some(params) => {
                    info!("There is already a cached entry in cache b_g2");
                    let b_g2_raw = params.clone();
                    unsafe {
                        let b_g2 = mem::transmute::<((Arc<Vec<<Bls12 as Engine>::G2Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G2Affine>>, usize)),
                            (Self::G2Builder, Self::G2Builder)>(b_g2_raw);
                        Ok(b_g2)
                    }
                },
                None => {
                    info!("nothing cached, build new cache b_g2");
                    let b_g2 = self.get_b_g2(num_inputs, _num_b_g2).unwrap();
                    unsafe {
                        let b_g2_raw = mem::transmute::<(Self::G2Builder, Self::G2Builder),
                            ((Arc<Vec<<Bls12 as Engine>::G2Affine>>, usize), (Arc<Vec<<Bls12 as Engine>::G2Affine>>, usize))>(b_g2.clone());
                        params_b_g2_cache.insert(key, b_g2_raw);
                    }
                    Ok(b_g2)
                }
            }
        }
    }
}

// A re-usable method for parameter loading via mmap.  Unlike the
// internal ones used elsewhere, this one does not update offset state
// and simply does the cast and transform needed.
pub fn read_g1<E: MultiMillerLoop>(
    mmap: &Mmap,
    range: Range<usize>,
    checked: bool,
) -> Result<E::G1Affine, std::io::Error> {
    let ptr = &mmap[range];
    // Safety: this operation is safe, because it's simply
    // casting to a known struct at the correct offset, given
    // the structure of the on-disk data.
    let repr = unsafe {
        &*(ptr as *const [u8] as *const <E::G1Affine as UncompressedEncoding>::Uncompressed)
    };

    let affine: E::G1Affine = {
        let affine_opt = if checked {
            E::G1Affine::from_uncompressed(repr)
        } else {
            E::G1Affine::from_uncompressed_unchecked(repr)
        };

        Option::from(affine_opt)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not on curve"))
    }?;

    if affine.is_identity().into() {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "point at infinity",
        ))
    } else {
        Ok(affine)
    }
}

// A re-usable method for parameter loading via mmap.  Unlike the
// internal ones used elsewhere, this one does not update offset state
// and simply does the cast and transform needed.
pub fn read_g2<E: MultiMillerLoop>(
    mmap: &Mmap,
    range: Range<usize>,
    checked: bool,
) -> Result<E::G2Affine, std::io::Error> {
    let ptr = &mmap[range];
    // Safety: this operation is safe, because it's simply
    // casting to a known struct at the correct offset, given
    // the structure of the on-disk data.
    let repr = unsafe {
        &*(ptr as *const [u8] as *const <E::G2Affine as UncompressedEncoding>::Uncompressed)
    };

    let affine: E::G2Affine = {
        let affine_opt = if checked {
            E::G2Affine::from_uncompressed(repr)
        } else {
            E::G2Affine::from_uncompressed_unchecked(repr)
        };

        Option::from(affine_opt)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not on curve"))
    }?;

    if affine.is_identity().into() {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "point at infinity",
        ))
    } else {
        Ok(affine)
    }
}
