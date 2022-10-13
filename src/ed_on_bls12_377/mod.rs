pub use ark_bls12_377::Fr as Felt;
pub use ark_ff::BigInteger256;
use ark_ff::Field;

mod sbox;

/// An instantiation of Anemoi with state width 2 and
/// rate 1 aimed at providing 128 bits security.
pub mod anemoi_2_1;

/// An instantiation of Anemoi with state width 4 and
/// rate 3 aimed at providing 128 bits security.
pub mod anemoi_4_3;

/// An instantiation of Anemoi with state width 8 and
/// rate 7 aimed at providing 128 bits security.
pub mod anemoi_8_7;

/// An instantiation of Anemoi with state width 12 and
/// rate 11 aimed at providing 128 bits security.
pub mod anemoi_12_11;

// HELPER FUNCTION
// ================================================================================================

#[inline(always)]
fn mul_by_generator(x: &Felt) -> Felt {
    let x2 = x.double();
    let x4 = x2.double();
    let x8 = x4.double();
    let x16 = x8.double();

    x16 + x4 + x2
}
