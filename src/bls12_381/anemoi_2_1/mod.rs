//! Implementation of the Anemoi permutation

use super::{mul_by_generator, sbox, BigInteger384, Felt};
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

/// Function state is set to 2 field elements or 96 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (48-bytes) is returned as digest.
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
                Felt::new(BigInteger384([
                    0x9bd2b150036616e3,
                    0xb76d6fcbb3177f14,
                    0x3993c83d93ea49c1,
                    0x3e809b73f9e764ae,
                    0x205437b9d9f901ec,
                    0x0ed7929b79fc8065,
                ])),
                Felt::new(BigInteger384([
                    0x44b4b57564527293,
                    0x6dd93f26404df096,
                    0x1d2e98ba6f461918,
                    0xad4ead0ab3c462d4,
                    0x85ddd24695619135,
                    0x153445d7cbe1f68b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x52a7073c65678c6f,
                    0x19b9edd7bb180718,
                    0xc27d08019672a93a,
                    0x7f1a5ac27c406851,
                    0x7b0e144d9ed5b9e0,
                    0x155da8b8da39a25d,
                ])),
                Felt::new(BigInteger384([
                    0xbfab6287a32a6057,
                    0x32983d215f92ed0f,
                    0xa53a37c5031d1ca4,
                    0x46debf228b883a47,
                    0xedc9d1d678f81a3e,
                    0x17e63bafcb7dcd78,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x115f9c3b1b3d1e07,
                    0xa458f6079f2de238,
                    0x66b9a2734ef53d20,
                    0x70b3d5a1abc9e631,
                    0xe9bbd0b444406979,
                    0x00005975ec0f4375,
                ])),
                Felt::new(BigInteger384([
                    0x8416adcf7395f8e3,
                    0x1af6b1a0da27760b,
                    0x78cea21cf791e41d,
                    0x8077ae003b290ffd,
                    0xeec2219742525be1,
                    0x0dac616679e778fe,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x180ed4eed77a3555,
                    0xf809736b74549bf7,
                    0xa7e4a88c24618cb7,
                    0x8df708fd71502416,
                    0x6ecd59706c048bdc,
                    0x04f3436e1c92d8eb,
                ])),
                Felt::new(BigInteger384([
                    0x5f69e83a6db2039f,
                    0x6ea94b6031520d09,
                    0x5fa97a9a593e18fa,
                    0x423737f3b94f4e08,
                    0x4d89272758007e02,
                    0x023d54c67f09c362,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3a1e48f763a14756,
                    0xca3931efea2fb075,
                    0x2a630db5cdf28023,
                    0xd662c70c1a87e179,
                    0xfc68959a540463fb,
                    0x1586e680e0240ceb,
                ])),
                Felt::new(BigInteger384([
                    0x45e83cb9a185d14a,
                    0x44efe81349bc9d9a,
                    0xf01c80e4a7b9b807,
                    0x5c23d9ea6295f8a7,
                    0x310d73eea46e939d,
                    0x1718542395669097,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x78de57d92bb3656c,
                    0xa408e40c819b9c6c,
                    0xbdc5feb686c09c09,
                    0x2cda32afd3a90599,
                    0x80739efe7216c37c,
                    0x0049efc39ce60bd0,
                ])),
                Felt::new(BigInteger384([
                    0xb30f01034af21062,
                    0xdb5a7c4c1b748300,
                    0x9f2c436abea67f80,
                    0x620d7f3289a3d86a,
                    0xa7b1d09b7393db12,
                    0x0c13573b7fcd7e3c,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger384([
                    0x1804000000015554,
                    0x855000053ab00001,
                    0x633cb57c253c276f,
                    0x6e22d1ec31ebb502,
                    0xd3916126f2d14ca2,
                    0x17fbb8571a006596,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger384([
                    0xf64900000018553d,
                    0x40f4005f6f0c0013,
                    0x9313f019a789cfb3,
                    0x59fb77168f0da76b,
                    0x951d2d06cf6bb694,
                    0x15b1e4359a873e00,
                ])),
                Felt::new(BigInteger384([
                    0x321300000006554f,
                    0xb93c0018d6c40005,
                    0x57605e0db0ddbb51,
                    0x8b256521ed1f9bcb,
                    0x6cf28d7901622c03,
                    0x11ebab9dbb81e28c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x221a96dee879506e,
                    0xf474857d61fd8d8e,
                    0x0cb62c7142732926,
                    0xdab2c2b279e8e05a,
                    0x66bfd291dacd2598,
                    0x14a7b8b22c4d8158,
                ])),
                Felt::new(BigInteger384([
                    0x8501827008bb463c,
                    0xe000d5f0f9385e8d,
                    0x651c1aea973d56a5,
                    0xc16f501811680aeb,
                    0xbd0d7da3633e27ed,
                    0x100d54ca211c6624,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4c220000000b554a,
                    0xed28002c72d80009,
                    0x4b84069f3c7f4f33,
                    0xa827f857a8538294,
                    0x0653b9cb0ff30b64,
                    0x0bdb9ee45d035f82,
                ])),
                Felt::new(BigInteger384([
                    0x43f5fffffffcaaae,
                    0x32b7fff2ed47fffd,
                    0x07e83a49a2e99d69,
                    0xeca8f3318332bb7a,
                    0xef148d1ea0f4c069,
                    0x040ab3263eff0206,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe0bdc1c5e2b55e06,
                    0xb76f0676228e5e74,
                    0x2fbaf59f66d056ea,
                    0xc8ff9cd89ca99aab,
                    0x6dbf68621e347688,
                    0x0fed36e07d855e85,
                ])),
                Felt::new(BigInteger384([
                    0x5ae7300061178397,
                    0x8428e8ffdc7a4d49,
                    0xa5bf09c57317f5ea,
                    0x8dd74862b2f69511,
                    0xe4d2dbc19d0eb2b2,
                    0x1008b4eaf6509d29,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x160f76e4efc25df3,
                    0x175d26ebfec0c569,
                    0x408ba2a98dd3b39d,
                    0x9d64978ade27493b,
                    0x68800917669696ca,
                    0x1282988896d35887,
                ])),
                Felt::new(BigInteger384([
                    0x9132f43c7c83b5f5,
                    0xfed8a739bfc5bef5,
                    0x043f4681f3c2d76e,
                    0x46faa5896847cd80,
                    0x9bd8fafba64bfff8,
                    0x15c744284d70fcf7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x00c28cc370ccfcf2,
                    0xbf600b6790d8a416,
                    0x8968a2a8eafec92f,
                    0xbfeb206d0b865c48,
                    0x65ca908dbe272ff8,
                    0x16f17ebc0b0e855d,
                ])),
                Felt::new(BigInteger384([
                    0x96627d39a1935a2a,
                    0xb7e6a3f8984bf689,
                    0xed47ba799c513e4f,
                    0xce0cf3518393b2f4,
                    0xc0a12a9de5b516fd,
                    0x02c7e79dcfb708ae,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x43d4099570773d56,
                    0x52e92fc67a472e1a,
                    0xaf0c4aeacf877dde,
                    0x5340b9ec5bcd20e7,
                    0x2bb0f3281ce398dc,
                    0x02b70cee9bc3cc41,
                ])),
                Felt::new(BigInteger384([
                    0xb2a267fddfd8f2d6,
                    0xa7999e6f67f9d13f,
                    0x0b88339bec563806,
                    0xef341415970f3e8f,
                    0xcf2d5c8d51952936,
                    0x184dd74a3fe8ce0b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x095181e5fa3564c7,
                    0x9efceb44ab69c43f,
                    0xa2154426dda44dfb,
                    0xd07e1e4f951bad64,
                    0xbece3bb9dc259c07,
                    0x0bc79fa1f7322d67,
                ])),
                Felt::new(BigInteger384([
                    0xac8436e24922461f,
                    0x924ca1f2968934d7,
                    0x0ec854e3c50451e0,
                    0x28f81694a4417f15,
                    0x04be49314253ba3c,
                    0x166433f5ce9b0657,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3751fca2bd3a713a,
                    0x8d7cb0cfc201afd1,
                    0xbc3ce9f65ad964e0,
                    0x2b90dddb359a45f7,
                    0x3d73e02585cf9fb3,
                    0x0c4b0359c599c669,
                ])),
                Felt::new(BigInteger384([
                    0x659d353f457b839c,
                    0x4795b198763e1729,
                    0xc8d795a0b7283136,
                    0x1d4980066ae6da07,
                    0x94fec801f7f16d39,
                    0x02bf12d9430c8d1c,
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
