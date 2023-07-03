//! Implementation of the Anemoi permutation

use super::{mul_by_generator, sbox, BigInteger256, Felt};
use crate::{Jive, Sponge};
use ark_ff::{Field, One, Zero};
use unroll::unroll_for_loops;

/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;

/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;
pub use hasher::AnemoiHash;

// ANEMOI CONSTANTS
// ================================================================================================

/// Function state is set to 2 field elements or 64 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 21 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 21;

// HELPER FUNCTIONS
// ================================================================================================

/// Applies the Anemoi S-Box on the current
/// hash state elements.
#[inline(always)]
pub(crate) fn apply_sbox_layer(state: &mut [Felt; STATE_WIDTH]) {
    let mut x: [Felt; NUM_COLUMNS] = state[..NUM_COLUMNS].try_into().unwrap();
    let mut y: [Felt; NUM_COLUMNS] = state[NUM_COLUMNS..].try_into().unwrap();

    x.iter_mut().enumerate().for_each(|(i, t)| {
        let y2 = y[i].square();
        *t -= mul_by_generator(&y2);
    });

    let mut x_alpha_inv = x;
    x_alpha_inv
        .iter_mut()
        .for_each(|t| *t = sbox::exp_inv_alpha(t));

    y.iter_mut()
        .enumerate()
        .for_each(|(i, t)| *t -= x_alpha_inv[i]);

    x.iter_mut().enumerate().for_each(|(i, t)| {
        let y2 = y[i].square();
        *t += mul_by_generator(&y2) + sbox::DELTA;
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
#[inline(always)]
pub(crate) fn apply_linear_layer(state: &mut [Felt; STATE_WIDTH]) {
    state[1] += state[0];
    state[0] += state[1];
}

// ANEMOI PERMUTATION
// ================================================================================================

/// Applies an Anemoi permutation to the provided state
#[inline(always)]
#[unroll_for_loops]
pub(crate) fn apply_permutation(state: &mut [Felt; STATE_WIDTH]) {
    for i in 0..NUM_HASH_ROUNDS {
        apply_round(state, i);
    }

    apply_linear_layer(state)
}

/// Applies an Anemoi round to the provided state
#[inline(always)]
#[unroll_for_loops]
pub(crate) fn apply_round(state: &mut [Felt; STATE_WIDTH], step: usize) {
    state[0] += round_constants::C[step % NUM_HASH_ROUNDS];
    state[1] += round_constants::D[step % NUM_HASH_ROUNDS];

    apply_linear_layer(state);
    apply_sbox_layer(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x549cb1b12215b5f8,
                    0xb9320b7623080e0a,
                    0xe21f92850bed29b5,
                    0x00e7b83bf28e88f3,
                ])),
                Felt::new(BigInteger256([
                    0xc46849152c8bfa9e,
                    0x271a9e7cd4417d5c,
                    0xcf6c148b5c10c200,
                    0x3549b83894f08391,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x925f58c06480019d,
                    0x853a386fa7586ab2,
                    0x5f61c869a872aa48,
                    0x2c5a820518063f94,
                ])),
                Felt::new(BigInteger256([
                    0xf656c806584ea0d6,
                    0x5d7992593311d429,
                    0xd592dacf9b683056,
                    0x3d0d7d18565e34e5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xaac25d6464647e5b,
                    0xd56f947ac4734b24,
                    0xd94de15d25bb2820,
                    0x24be138a097e25db,
                ])),
                Felt::new(BigInteger256([
                    0x0dcc53187ca26b4a,
                    0x0e11e5386c6854d4,
                    0x9b66013dc4fc5a71,
                    0x02f11432fa70dc89,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd82affcb320a6cf0,
                    0x99ba4e4af06d5824,
                    0x847b0a3c9b874593,
                    0x2b8a60896dbedfc6,
                ])),
                Felt::new(BigInteger256([
                    0x4513d7dc197ec40a,
                    0xdbfdaafb008f2527,
                    0x37fed91c601c809e,
                    0x0513408c3ab1f33c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0949782eb779d666,
                    0x06c9ebfb395c1b7b,
                    0xbfb48908817e089d,
                    0x270579da9f12bb55,
                ])),
                Felt::new(BigInteger256([
                    0x691e725ea3f8a0a6,
                    0x1ca4640964701e25,
                    0x9db9d4a52f2a1073,
                    0x15a102922fd2eb46,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x880576e13cc90b1f,
                    0x68e9be46608cbe1f,
                    0x21699f5b3006d2aa,
                    0x3a4cc129c56b6261,
                ])),
                Felt::new(BigInteger256([
                    0xe6c770f10c7cc0c7,
                    0x0909c7d04855b38e,
                    0x9b3c7d1960ebdf46,
                    0x117da9beb0d61a5b,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0x0a7e7c3e99999999,
                    0xb83c0a9bfa6b6a89,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x3785babe3451890e,
                    0xbad6a04b96e7e335,
                    0xdaecef0c6271ae56,
                    0x309b6fa4569ce2e5,
                ])),
                Felt::new(BigInteger256([
                    0x376eae911f9c7044,
                    0xf15d13a83fed92e2,
                    0x8a6f1eb3ded38974,
                    0x0a63a5046381f69e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe89d156895f33598,
                    0x22cc8f90197cdefa,
                    0x59d1af10ece372fe,
                    0x3c38ccf4eeef1fb8,
                ])),
                Felt::new(BigInteger256([
                    0x1fae7c1b952b2763,
                    0x6ff92987f72625f6,
                    0x05ce29c9b82e2059,
                    0x02ee5a311dcf8a77,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xae41e60699999981,
                    0x819db2fb1b340ff2,
                    0xccccccccccccccc9,
                    0x0ccccccccccccccc,
                ])),
                Felt::new(BigInteger256([
                    0x64b4c3b400000004,
                    0x891a63f02533e46e,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x91036227f225de02,
                    0x55350fa37393df7a,
                    0x245fa221cf61c197,
                    0x123cf58f73eab51a,
                ])),
                Felt::new(BigInteger256([
                    0xd63d2cac361d05f1,
                    0xb7abe04b340581cb,
                    0x1e2782bab31ed630,
                    0x18a4ff9a32b13d3f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xac651fe78adb1249,
                    0xe9bd87b589846a1a,
                    0x854662239be6242e,
                    0x276fa8c10b1c97d6,
                ])),
                Felt::new(BigInteger256([
                    0xc192ab7fb075f32a,
                    0x8aad2735f4327278,
                    0xadb89774fbc2b737,
                    0x12662c0015bba883,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfdc9f7e0c891be47,
                    0x4a6f2de9d31dc82d,
                    0xa3579fd2dc392bba,
                    0x0afb7169ade45919,
                ])),
                Felt::new(BigInteger256([
                    0xf64184aa8e34742a,
                    0x13a822cc38a571f8,
                    0x354c482528f08ca8,
                    0x04fd9e24b40a1e11,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb47e1194546aed69,
                    0x41ad3e7d8a2304ea,
                    0x6bfb4210c3dd4910,
                    0x03161b862d3eb871,
                ])),
                Felt::new(BigInteger256([
                    0xb6e68fc09612048e,
                    0x1b0540f4cab4ed0c,
                    0x95c38b7e41232183,
                    0x3dfcafc421e4c0a6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdbfb9c57a4cf8f94,
                    0xb4f45f76d45c0011,
                    0xad4c8f76b5019584,
                    0x09da6dfe7ccdf2e7,
                ])),
                Felt::new(BigInteger256([
                    0xdd886bfb79a8652f,
                    0x5128fe3d83d7c981,
                    0x06748c0ef3cd7099,
                    0x382b675fa0fde2a4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf4ea799842a8cad4,
                    0x64e465740ceea336,
                    0x13bb704e5b3e577c,
                    0x2773b1b165690126,
                ])),
                Felt::new(BigInteger256([
                    0x04395e5734733d92,
                    0x708793e22b3563d0,
                    0x3cd0f208bad77d5a,
                    0x23141077d2678182,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            apply_sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
