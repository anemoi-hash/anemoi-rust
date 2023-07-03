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

/// Function state is set to 6 field elements or 288 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 6;
/// 5 elements of the state are reserved for rate.
pub const RATE_WIDTH: usize = 5;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 3;

/// One element (48-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = 1;

/// The number of rounds is set to 12 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 12;

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
    apply_linear_layer_internal(&mut state[..NUM_COLUMNS]);
    state[NUM_COLUMNS..].rotate_left(1);
    apply_linear_layer_internal(&mut state[NUM_COLUMNS..]);

    // PHT layer
    state[3] += state[0];
    state[4] += state[1];
    state[5] += state[2];

    state[0] += state[3];
    state[1] += state[4];
    state[2] += state[5];
}

#[inline(always)]
fn apply_linear_layer_internal(state: &mut [Felt]) {
    let tmp = state[0] + mul_by_generator(&state[2]);
    state[2] += state[1];
    state[2] += mul_by_generator(&state[0]);

    state[0] = tmp + state[2];
    state[1] += tmp;
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
    // determine which round constants to use
    let c = &round_constants::C[step % NUM_HASH_ROUNDS];
    let d = &round_constants::D[step % NUM_HASH_ROUNDS];

    for i in 0..NUM_COLUMNS {
        state[i] += c[i];
        state[NUM_COLUMNS + i] += d[i];
    }

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
            [Felt::zero(); 6],
            [Felt::one(); 6],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            [
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger384([
                    0x563ac35ef02d3f03,
                    0x2d0607afc3f647ef,
                    0x0c9cdb0e74c52937,
                    0xd5ac7c510b99cbf3,
                    0xddc1628e1f069305,
                    0x07e52766f1b90f0f,
                ])),
                Felt::new(BigInteger384([
                    0x558ec80d4273b849,
                    0xd1efd057ece462c9,
                    0x069f6c91e904f07f,
                    0x1cd4db654d1b9bb0,
                    0xcd8a5869d8182fda,
                    0x001923c9cf266711,
                ])),
                Felt::new(BigInteger384([
                    0x08f55864efc57841,
                    0xa14a51fb325f1b89,
                    0x2942fe78427a4f5a,
                    0x5621a998d6852dd1,
                    0xedd49609b04a74af,
                    0x1266f451155b0ccf,
                ])),
                Felt::new(BigInteger384([
                    0x6686c6b919db640a,
                    0xdc0ae7b3151d3a98,
                    0x762c5cfc3642cc30,
                    0xb56f91e9ba11a24a,
                    0xaf1dbee871cd4651,
                    0x0587f93d8c2eebb8,
                ])),
                Felt::new(BigInteger384([
                    0x58ebcc50f3697227,
                    0x0a9c4867b3647e03,
                    0x35fb27c03e2c010f,
                    0xc1057f065ada2868,
                    0xe4145233a8798f50,
                    0x0742c8011a380e01,
                ])),
                Felt::new(BigInteger384([
                    0x743b93715ae5c5fa,
                    0xe5c84ebb440bcee6,
                    0x87acf1bcdba5900a,
                    0x6315c04fbb8fa04c,
                    0xa65efb51acdf8648,
                    0x06ceb04a0563ef42,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x22257ee2a27b509e,
                    0xf6d34888d3126651,
                    0xd2977dac7d4114d2,
                    0x685f4b65584f4002,
                    0xc596f0eb3a9d8281,
                    0x0120caadf4176872,
                ])),
                Felt::new(BigInteger384([
                    0x57e1624ccf1050d1,
                    0xca78ce89fe0d9f95,
                    0x7f485d01f841c04e,
                    0xf277ae38a94a35ed,
                    0xadf4184f829b6cda,
                    0x018066b8b9a61729,
                ])),
                Felt::new(BigInteger384([
                    0x28979c91695d5b46,
                    0xb5a0b1583c885b83,
                    0xf5da9b2e4e90e218,
                    0x5657c0a17c9cfcc6,
                    0x04cc15c3e0dd5bd5,
                    0x04d4675955bd35e6,
                ])),
                Felt::new(BigInteger384([
                    0xacd8c8c0748a89c1,
                    0x927e5bb701a7e218,
                    0x87d08cfc831107ad,
                    0x3feccd3e4507c44e,
                    0x39eb03cbf9f0aed2,
                    0x0de55d6dec40c9d6,
                ])),
                Felt::new(BigInteger384([
                    0x1833171c7ab4b7f5,
                    0x602356e9a2bed570,
                    0xbd6202894fda9e0b,
                    0x6a1e79eb0c9c2941,
                    0x323252cdf907d759,
                    0x0058e512c09c0ae4,
                ])),
                Felt::new(BigInteger384([
                    0xab591e647e1d3da3,
                    0x77bf92ac4e9c25b0,
                    0xd74ca9afb25bb85c,
                    0xf175a82fa672b9d7,
                    0x72470362414de14d,
                    0x14e4f098f576e249,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6a085cc68c5160b3,
                    0x2723b5baafd29f80,
                    0xdb45d33db2ff3308,
                    0x41f2cd74dfa41060,
                    0x24a958ffb99ce2e7,
                    0x0d35ddf2eed70916,
                ])),
                Felt::new(BigInteger384([
                    0xd7ad1ff0269bb796,
                    0x6743626900058a92,
                    0x0432a92436b93855,
                    0x908609687183cde8,
                    0xa8771d9bb274a4ae,
                    0x0db8b688bb32dad8,
                ])),
                Felt::new(BigInteger384([
                    0x4d56ba8b64d65520,
                    0xa961b74fbcabed0a,
                    0x257d084417714622,
                    0xcb72ab016b68746c,
                    0xed5e990ed3907a01,
                    0x143136155bd5b9f9,
                ])),
                Felt::new(BigInteger384([
                    0x154bb720b6e2192e,
                    0xa8ce187dd7e0ffd8,
                    0x4a76938b7889096c,
                    0x36a8650bf9723359,
                    0x74c57bb57ff8121f,
                    0x0e65d3be8f2a2e99,
                ])),
                Felt::new(BigInteger384([
                    0x67be8b3ccad4257b,
                    0xe3a4203e0eba4e4b,
                    0x0d51812402672031,
                    0x5380e16e7f630e11,
                    0xa139cbc7bbbe96eb,
                    0x1732ae4c2cc27930,
                ])),
                Felt::new(BigInteger384([
                    0x0806107bb612876a,
                    0xde2085cb1216d7ce,
                    0x209e9943facb66a5,
                    0xdc06a17d5442d71d,
                    0x4153475a8cab786d,
                    0x05bbc58635620819,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x74ad684e3aefb3f1,
                    0x2d4d25b34595d2f3,
                    0x3c465fc367d49cfa,
                    0xebe468491b558d66,
                    0x30ededa1228d8f7e,
                    0x1792be9dbb4d4c5d,
                ])),
                Felt::new(BigInteger384([
                    0x36074110f4ca083b,
                    0x7204d6526500eeb6,
                    0xd3892b0705b1dd95,
                    0x0cfc420c45d947c1,
                    0x4d850860150e937a,
                    0x098e5b4ca7821144,
                ])),
                Felt::new(BigInteger384([
                    0xb2cad6e0e98e99b9,
                    0xc1ddad45223e9f09,
                    0xb6e01bae4f3e80a3,
                    0xa80849baa67d9f00,
                    0x7d545be36d20a224,
                    0x08d46daeacbbc563,
                ])),
                Felt::new(BigInteger384([
                    0x49f7c20d48b07300,
                    0x73d81713d55ca0cb,
                    0x545cc60aacc9c802,
                    0x88452304c81011e0,
                    0x75d81286ce9ffc1b,
                    0x04e23f59ad65d31c,
                ])),
                Felt::new(BigInteger384([
                    0xa3663cc58a3d298a,
                    0x9a31695c365790f2,
                    0xc048aa311b8f761e,
                    0x8b828259e9fe4e4e,
                    0x345298d6e2b005d9,
                    0x0a71eff35fb2e414,
                ])),
                Felt::new(BigInteger384([
                    0x1f658074efb99db1,
                    0xeebaa2c06bdc133b,
                    0xa0a61739f6c9fa14,
                    0x7287aaec7dbf6a59,
                    0x9616cdc3fa1ea7d3,
                    0x0c9ed65bb7b2ef36,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3e8e76051c10bd29,
                    0x0f9c4366b3fea533,
                    0xe9903e772670dbce,
                    0x9b7d72d21e05240f,
                    0x11db5e3b43ddaf90,
                    0x041a35dbdb884f72,
                ])),
                Felt::new(BigInteger384([
                    0x4aaa0cf65ee221fd,
                    0xea67d872ea9e438a,
                    0xd6edb017ddfb742f,
                    0x64ba5b494bba1c3f,
                    0x314128641041088b,
                    0x10614c351edd16e1,
                ])),
                Felt::new(BigInteger384([
                    0x31d288023795a2e7,
                    0xdd0c766fe33e35b6,
                    0x436a508e8e3ff4ec,
                    0xb050c468ba2c2cd6,
                    0x9b825297a7fd1c62,
                    0x076d49bbe774bc69,
                ])),
                Felt::new(BigInteger384([
                    0x060f3cd1db16cd74,
                    0x25b1fab8fd8da43e,
                    0x2d815f175dfcf61c,
                    0x1cb52c7095e9c5d0,
                    0x99cd64f78a560f4a,
                    0x0bb6636574de5a36,
                ])),
                Felt::new(BigInteger384([
                    0x412fe8e523bd5e4c,
                    0x77c209ad388bbd9c,
                    0x71cec589b22be382,
                    0xb12774123d7decb4,
                    0x5707d331ca01b7bf,
                    0x19a620e43d48415a,
                ])),
                Felt::new(BigInteger384([
                    0x67b5e0feda522998,
                    0x7be80d41d40c0256,
                    0xcfeb089619741d39,
                    0x48743728ab8c08a6,
                    0x05f7c1eb704d2439,
                    0x0d2dd33bfc04bb50,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xf31bccdd843d16f6,
                    0xad2181b41bec87f5,
                    0xddb34cbe3dfc8caa,
                    0x12329a1a065edaee,
                    0xda44d8e782fbb5d3,
                    0x0e962c483de8092e,
                ])),
                Felt::new(BigInteger384([
                    0xb837d4cf10c2bdf8,
                    0x6cd7bddaafdc2e0a,
                    0xcbc0411d1699dde3,
                    0x8e0aed11b9787b86,
                    0xfcd6bb05e83fb7f9,
                    0x06f71d8ca7d8166e,
                ])),
                Felt::new(BigInteger384([
                    0x963ac22280103d66,
                    0x693534188c6ef454,
                    0x155b618ed48f3096,
                    0xfeec79d1a0c764b3,
                    0x74adf32b97e4fde4,
                    0x043aadb14525f31c,
                ])),
                Felt::new(BigInteger384([
                    0x281c1d39dcba1583,
                    0x045ddf1a64b57473,
                    0x4374146f110dc061,
                    0xb99752f83e097d4b,
                    0xbc4a34d1b493bc53,
                    0x008fe7562fb7d706,
                ])),
                Felt::new(BigInteger384([
                    0xa9073d803e7da22c,
                    0x50e138403f3afef5,
                    0x0ac05936d24fd242,
                    0x41e9be9b2872ac4b,
                    0x408c400edd622802,
                    0x0e0048a4a89728f6,
                ])),
                Felt::new(BigInteger384([
                    0x5e975db03dd26e3d,
                    0xc08a4f418d8ad2a8,
                    0x148a51376d573b97,
                    0x04b4bcc0033b66ac,
                    0xcf8caeed02d3f1c9,
                    0x073f9ff1fdd66909,
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
                Felt::new(BigInteger384([
                    0x1804000000015554,
                    0x855000053ab00001,
                    0x633cb57c253c276f,
                    0x6e22d1ec31ebb502,
                    0xd3916126f2d14ca2,
                    0x17fbb8571a006596,
                ])),
                Felt::new(BigInteger384([
                    0x1804000000015554,
                    0x855000053ab00001,
                    0x633cb57c253c276f,
                    0x6e22d1ec31ebb502,
                    0xd3916126f2d14ca2,
                    0x17fbb8571a006596,
                ])),
                Felt::zero(),
                Felt::zero(),
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
                    0xf64900000018553d,
                    0x40f4005f6f0c0013,
                    0x9313f019a789cfb3,
                    0x59fb77168f0da76b,
                    0x951d2d06cf6bb694,
                    0x15b1e4359a873e00,
                ])),
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
                Felt::new(BigInteger384([
                    0x321300000006554f,
                    0xb93c0018d6c40005,
                    0x57605e0db0ddbb51,
                    0x8b256521ed1f9bcb,
                    0x6cf28d7901622c03,
                    0x11ebab9dbb81e28c,
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
                    0x221a96dee879506e,
                    0xf474857d61fd8d8e,
                    0x0cb62c7142732926,
                    0xdab2c2b279e8e05a,
                    0x66bfd291dacd2598,
                    0x14a7b8b22c4d8158,
                ])),
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
                Felt::new(BigInteger384([
                    0x8501827008bb463c,
                    0xe000d5f0f9385e8d,
                    0x651c1aea973d56a5,
                    0xc16f501811680aeb,
                    0xbd0d7da3633e27ed,
                    0x100d54ca211c6624,
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
                    0x4c220000000b554a,
                    0xed28002c72d80009,
                    0x4b84069f3c7f4f33,
                    0xa827f857a8538294,
                    0x0653b9cb0ff30b64,
                    0x0bdb9ee45d035f82,
                ])),
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
                Felt::new(BigInteger384([
                    0x43f5fffffffcaaae,
                    0x32b7fff2ed47fffd,
                    0x07e83a49a2e99d69,
                    0xeca8f3318332bb7a,
                    0xef148d1ea0f4c069,
                    0x040ab3263eff0206,
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
                    0x0133ebe3fae693d3,
                    0xbdbfaaa58acd76c4,
                    0x90215d15854015f6,
                    0xb95d0e489d2afabe,
                    0x84c5d52786bcc459,
                    0x0a94c4354662febd,
                ])),
                Felt::new(BigInteger384([
                    0x8806c2121094a515,
                    0x8145798c3d41bb2e,
                    0xd3af6f27da460012,
                    0xf1278c767c2e297e,
                    0x0dae330e472a3f73,
                    0x0a338847fabd57ad,
                ])),
                Felt::new(BigInteger384([
                    0x4e2d38b895e70a0b,
                    0x3526014ca7e9d64c,
                    0x67b3f23732917bff,
                    0x7fbcaf79151a3c66,
                    0xbc7dae81c432fc9d,
                    0x07d9b35d09ba5e51,
                ])),
                Felt::new(BigInteger384([
                    0x7307f69cfa4771d1,
                    0xd831836ae0027002,
                    0xf45d263e10597b66,
                    0x4bc29b1de9953d5c,
                    0x70a08ea85c0dffcc,
                    0x16c941b574ffa0b8,
                ])),
                Felt::new(BigInteger384([
                    0x73971f56f47c58d7,
                    0xacc86d58b368ce5e,
                    0x2d2d5cb841867d6b,
                    0x25247d778e9f1966,
                    0x6f59d7c7c591e8d6,
                    0x101a4447c5288a0a,
                ])),
                Felt::new(BigInteger384([
                    0x18315037ae8b58ca,
                    0x94f06069dbf4726a,
                    0x4805306121a8b7df,
                    0x5bf29877824272f8,
                    0x079a8eccc23456a2,
                    0x10978cf31f8c257e,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb244bc2c91dae6b7,
                    0x9946ae9a6463abc3,
                    0x3531504a9c9f1c45,
                    0xbcdbba461a93cbd4,
                    0x83f0706c7b02d737,
                    0x140598f866dd8e7f,
                ])),
                Felt::new(BigInteger384([
                    0xf42768533c41b208,
                    0x9f5913a0f425432e,
                    0x748767d794229cec,
                    0xa4325a576ed4a2bf,
                    0xd22b4013529f13e4,
                    0x0c55d79fcc90af48,
                ])),
                Felt::new(BigInteger384([
                    0xc7a0a4143db1db5a,
                    0x7d8d7222b94b175f,
                    0x0e22cbfc72238f22,
                    0xc6cbcb8055b261f8,
                    0x6c47a5ecfd00f75d,
                    0x067ec4c3146449ad,
                ])),
                Felt::new(BigInteger384([
                    0x4911fd620b2ddf6e,
                    0xe939e76f0902d90f,
                    0xf5c065115625de90,
                    0xc43a731e5ee1a8a5,
                    0xacbe077138183dfd,
                    0x0c927bf6a3441e5f,
                ])),
                Felt::new(BigInteger384([
                    0xd55c15fbcb3b639a,
                    0x77b36d3aedfc5d26,
                    0xb38ffbb9d4c08aee,
                    0xfef3e85d221fa73d,
                    0x6d0f778c8ebac785,
                    0x1748103ba88f6ae3,
                ])),
                Felt::new(BigInteger384([
                    0x3c3053c1c86ca33f,
                    0xd95afea8292e4384,
                    0xc0fce93a3b2bead6,
                    0xe76192faeb16a446,
                    0x8103d13196d838a0,
                    0x0e0942dd447322f9,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2f580bdef8bbf83f,
                    0xa194e0c0aafda077,
                    0xcfd36eb984183f06,
                    0xf520791991d51009,
                    0xddd91f1a48d59773,
                    0x003fe31077deddc2,
                ])),
                Felt::new(BigInteger384([
                    0x81640c8aa9916873,
                    0x867c3ed29069aebf,
                    0x36b2f1113b2e2f01,
                    0xd8fbbccea7b98f63,
                    0x9cbbd1b896e1dd50,
                    0x134170d84647fc84,
                ])),
                Felt::new(BigInteger384([
                    0xa604232616362dd6,
                    0x582f23270da12eda,
                    0xe3f824787bec8690,
                    0x7ea00be3640e5b3d,
                    0x2443695126500060,
                    0x13188958b0547383,
                ])),
                Felt::new(BigInteger384([
                    0x86f201d72f64fa2e,
                    0x83b40c56c81b15a5,
                    0x0d8dc5c30ea3a0e9,
                    0xe4a5eb7bb12d1eca,
                    0xccea033e1567e9e2,
                    0x09c648e5861160b2,
                ])),
                Felt::new(BigInteger384([
                    0xc390bc46872e845a,
                    0xcb687c6032f55f1a,
                    0x954b61ecbda43666,
                    0x37c833e5eee005eb,
                    0xd42b63dcd5fc4363,
                    0x0d0c373533502c64,
                ])),
                Felt::new(BigInteger384([
                    0xd7ca057270871a2c,
                    0x2157b764e3a5a7a8,
                    0xa90551b840a0cbe9,
                    0xe43be53a1cfbc31a,
                    0x51694772ace68602,
                    0x07e78f33e1622dac,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xfcb7fcf33ea164c9,
                    0xa37da2447edd6f10,
                    0x1a9ee6db344f95f4,
                    0x94ba03850bd350f8,
                    0xd6688b05d82305cd,
                    0x0f07cc5c37391e76,
                ])),
                Felt::new(BigInteger384([
                    0x3e33ffc52824c9c5,
                    0xba5a5af061984297,
                    0xb1fb083cbd5c6f5d,
                    0xc90beb9620978d4a,
                    0x45cdede5590b3231,
                    0x12cf47bc596c498b,
                ])),
                Felt::new(BigInteger384([
                    0xa4995a4894f331a1,
                    0x3840045a5de9e27b,
                    0x715201422c14540d,
                    0xf56ebf86cd15e70f,
                    0x0d85c8a959a73356,
                    0x0d06ba0619f3b172,
                ])),
                Felt::new(BigInteger384([
                    0xc046ea22aea07935,
                    0x732280ae4037059f,
                    0x29670e870e15df61,
                    0xe5837d3610f5c97e,
                    0x108e35fc14cf8080,
                    0x1664648b8685d82e,
                ])),
                Felt::new(BigInteger384([
                    0xb015292686507517,
                    0x810162ee7c84c8f2,
                    0x9f4e0c6f0e1f4b25,
                    0xf692b20a6fde5685,
                    0xe97285212c188cfa,
                    0x051dd9a881b7886a,
                ])),
                Felt::new(BigInteger384([
                    0x52dfe86c1a8a3ab2,
                    0x3e323e134e759030,
                    0x7495cd2aa24447af,
                    0xc5df2112ebe55db5,
                    0xc3224598be203298,
                    0x0165e64c109d2cc5,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd6d4586be458c7f9,
                    0xd3ec2c93ac3e18a4,
                    0xef0476482403ed66,
                    0x9fd823765ed0c584,
                    0x92e71e2c7f730167,
                    0x0928d7cb714e59b7,
                ])),
                Felt::new(BigInteger384([
                    0xe9278c8729e036be,
                    0xa9ee8c2dd2eb4b0e,
                    0xad511170602aa9ab,
                    0x88766329d538002c,
                    0x38a132a901b73526,
                    0x08220b268edfc085,
                ])),
                Felt::new(BigInteger384([
                    0xfc4b131d1c46f7d1,
                    0x4b60bc830e991b01,
                    0x861c4c591f5601e9,
                    0x7c2ec4bcd55ac1d7,
                    0x38b48b428668f051,
                    0x0f64f2c7f5df3994,
                ])),
                Felt::new(BigInteger384([
                    0x3c8c9def8ae9be76,
                    0x085e8fed06587e2a,
                    0x6aae6b5635ad87d7,
                    0x292aa25f8bfec407,
                    0xfeffd2631b1e8792,
                    0x1033b26112dffb9b,
                ])),
                Felt::new(BigInteger384([
                    0xb487ec7d558e07cf,
                    0xdcd8d1f518788ab2,
                    0xee376c7a3235e161,
                    0x328b5f2cdf6b275f,
                    0xf6790cdd9d04da07,
                    0x0a785585243c7c2e,
                ])),
                Felt::new(BigInteger384([
                    0x969567bba94d1a43,
                    0x5d97d91cb322befe,
                    0xe802ce7734de4a83,
                    0x38a78abfce69dcf5,
                    0xd397f1ed04fd6247,
                    0x15600b5611cea001,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2baa5141235603da,
                    0x2d9a6adace930537,
                    0x4eb989e64d3add7c,
                    0x4daa0e6c2c4f7785,
                    0x72c8334df37ccf7f,
                    0x1962d52872f833b4,
                ])),
                Felt::new(BigInteger384([
                    0x9daa9452dad1480e,
                    0xb8889a615912c137,
                    0x52c676b0c0764d54,
                    0x99a0022f2152ef16,
                    0xb2411c0d7ce3d66d,
                    0x15d6baed7cb267c6,
                ])),
                Felt::new(BigInteger384([
                    0x774664e0d42cac6b,
                    0xbe7a3f60f1a718dc,
                    0x5675fb33479466cd,
                    0x8f13c3babefff5b6,
                    0x3dc1b7cf47830e13,
                    0x046cf353bb4c9598,
                ])),
                Felt::new(BigInteger384([
                    0x7305e5080690a857,
                    0x60e4ce361052099c,
                    0x639a8e8ab903a05b,
                    0x49fc1561e10a415c,
                    0x212288ae2934f28f,
                    0x032a92f3447ff0c2,
                ])),
                Felt::new(BigInteger384([
                    0x368931ce06babf19,
                    0xe39e7c3bb177514c,
                    0x871e84331a23a20c,
                    0xd52af00929f3259c,
                    0xde54b93ccddcd64f,
                    0x048abd2a3dde99a2,
                ])),
                Felt::new(BigInteger384([
                    0x6e23eb50cba32407,
                    0xdb98c2ecb51a218a,
                    0x1a9bd733dbea2fa4,
                    0x6ebc6d0e425d505e,
                    0x953894db2013e783,
                    0x02905871d27f1761,
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
