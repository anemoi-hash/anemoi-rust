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
                    0xd5a3a0426e05ffa5,
                    0xa5c0e9b03b5a7e1a,
                    0x73157f691d4d3157,
                    0x11442391220ef410,
                ])),
                Felt::new(BigInteger256([
                    0x465e7074d75a5aaa,
                    0x05bf5deced0bce35,
                    0x29fceb0a8cdee652,
                    0x1f7c476691873268,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x93131ba8e928ce3b,
                    0x8be409f70dde9733,
                    0xaa71f8205ca0c84c,
                    0x1d947c182e799959,
                ])),
                Felt::new(BigInteger256([
                    0x37e8b8fa4ff03b6b,
                    0x9ec8ef97d469d865,
                    0x99f846fbb4481552,
                    0x09b3d5ce9136bfa1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x358216ce9e40f47b,
                    0xd2cfd0313bc7ddff,
                    0x1293cd59110d04cc,
                    0x3ed6e48bf5fbe051,
                ])),
                Felt::new(BigInteger256([
                    0x41e7f2914b3356b3,
                    0x2b34973ba3132094,
                    0x7207263eee3dfc1e,
                    0x04d6eb43de5ab83e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbd484ac236505a28,
                    0x85c580c4e0c874ce,
                    0xa2ec9eec211f6574,
                    0x4b5fdbf77176b1da,
                ])),
                Felt::new(BigInteger256([
                    0x8387e91a874321d5,
                    0xdcda8ebfed618ad0,
                    0xb579288ad609c80f,
                    0x4cb98d07a691b982,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xad3961b75017230c,
                    0x8f1d43ac8017ecce,
                    0xcdffbef591643643,
                    0x009216424356a200,
                ])),
                Felt::new(BigInteger256([
                    0x76f340bf5e1cd124,
                    0xe07cb9ec0b4fe094,
                    0x626856345ae8fe33,
                    0x3c772f5221bb060d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9596acc5a15944e4,
                    0xf6e1da94c37df3ca,
                    0xfc1a7d5b9cdfcbcb,
                    0x51aceb76e011ffec,
                ])),
                Felt::new(BigInteger256([
                    0x58a4d7af1f8cf3c6,
                    0x947e8d7688ad9998,
                    0x23f8c439370a8d5b,
                    0x6c93113011f1ecb1,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0xdb6db6dadb6db6dc,
                    0xe6b5824adb6cc6da,
                    0xf8b356e005810db9,
                    0x66d0f1e660ec4796,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xd99f4a8fe00e11bd,
                    0xe12a753d82b33b30,
                    0xba87bf86c47e6f0b,
                    0x58de312b5993b67f,
                ])),
                Felt::new(BigInteger256([
                    0xb4c1972e044567f4,
                    0x8e14aacbecb9fb59,
                    0x2c0f182612f81cd9,
                    0x34d6a5db27797543,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x34413b9512617cf9,
                    0x7e6bdaab3f8443c5,
                    0x752178862c9a6ab7,
                    0x3dde7b14d2951772,
                ])),
                Felt::new(BigInteger256([
                    0xbdf2fa0b931e7a04,
                    0x60f39c6eae7a9a3c,
                    0x34e97cf1bd62a467,
                    0x13b9e8a487cc4745,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdb6db6ecdb6db6ca,
                    0x035ffa14db8a4eec,
                    0x5ea2264f581fdd5a,
                    0x401b2e0d73d97883,
                ])),
                Felt::new(BigInteger256([
                    0xfffffffd00000003,
                    0xfb38ec08fffb13fc,
                    0x99ad88181ce5880f,
                    0x5bc8f5f97cd877d8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd8911e9adc2ec2d4,
                    0xcdbd79157c65fd99,
                    0xa5cb8b544c3fcd32,
                    0x11afd11d7283fd0f,
                ])),
                Felt::new(BigInteger256([
                    0xe0382dafc0f2eeca,
                    0x6146b70e5a1bb1c7,
                    0x459620b48f22f6ae,
                    0x0401d65ed2d49c7d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x771302c3f5618040,
                    0x44c87b26d94f61a6,
                    0x5df2507d818d7823,
                    0x247c577dd2968a65,
                ])),
                Felt::new(BigInteger256([
                    0xf6175f39e682cc4e,
                    0x6c9bae5ebdea9b7d,
                    0x1171dcdd97d4b28a,
                    0x15a15f7cda4f90c5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd24b91b9db2bd552,
                    0x3c90578db43c9f69,
                    0x4ef2e0b8926cb5b2,
                    0x6aa6a393a2ac4ff3,
                ])),
                Felt::new(BigInteger256([
                    0x43ada4a7cb55a1a4,
                    0x8c7455cc741aff26,
                    0x91a542035898e505,
                    0x477ac2682c617946,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xda57c3b5fd3ebfa8,
                    0xc271ec687bc167f1,
                    0x06f4a5e0045352af,
                    0x38c2376f1f1dbe4e,
                ])),
                Felt::new(BigInteger256([
                    0xa8befe744bcf5772,
                    0x752ea4e73177ffe0,
                    0xcd67eeeec7e3931b,
                    0x201f11428d31af50,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2202a93f6178a838,
                    0x30c211f5fc9f6101,
                    0xca2e036d85c24ce9,
                    0x351d4f32c3fc5a79,
                ])),
                Felt::new(BigInteger256([
                    0x2aca99701913444b,
                    0x73a6d321e7f3fc51,
                    0x782ba539b63ca96f,
                    0x469a43d4446a2dd9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x998d0fcc3c409fa6,
                    0x47ca120a82cb7057,
                    0x870f5baf8fb82d6e,
                    0x303f01324c134473,
                ])),
                Felt::new(BigInteger256([
                    0x2510bc4e6f4bae4c,
                    0xf6967463ffa5162c,
                    0x7597c0dd31d223b9,
                    0x6b5d7fdb3e845b3d,
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
