//! This crate provides an implementation of different instantiations
//! of the Anemoi permutation, and applications to a Sponge mode and a
//! novel Jive compression mode.
//!
//! All hash instantiations are defined using an `Sponge` trait and can both
//! process sequences of bytes or native field elements.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

mod traits;
pub use traits::*;

/// An implementation of some instantiations of the Anemoi permutation
/// in Sponge and Jive modes over BLS12-377 base field.
#[cfg(feature = "bls_377")]
pub mod bls12_377;

/// An implementation of some instantiations of the Anemoi permutation
/// in Sponge and Jive modes over BLS12-381 base field.
#[cfg(feature = "bls_381")]
pub mod bls12_381;

/// An implementation of some instantiations of the Anemoi permutation
/// in Sponge and Jive modes over BN-254 base field.
#[cfg(feature = "bn_254")]
pub mod bn_254;

/// An implementation of some instantiations of the Anemoi permutation
/// in Sponge and Jive modes over Jubjub base field.
#[cfg(feature = "jubjub")]
pub mod jubjub;

/// An implementation of some instantiations of the Anemoi permutation
/// in Sponge and Jive modes over Pallas base field.
#[cfg(feature = "pallas")]
pub mod pallas;
