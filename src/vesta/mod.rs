pub use ark_ff::BigInteger256;
use ark_ff::Field;
pub use ark_pallas::Fr as Felt;

mod sbox;

/// An instantiation of Anemoi with state width 2 and
/// rate 1 aimed at providing 128 bits security.
pub mod anemoi_2_1;

/// An instantiation of Anemoi with state width 4 and
/// rate 3 aimed at providing 128 bits security.
pub mod anemoi_4_3;

// HELPER FUNCTION
// ================================================================================================

#[inline(always)]
fn mul_by_generator(x: &Felt) -> Felt {
    let x2 = x.double();
    let x4 = x2.double();

    x4 + x
}
