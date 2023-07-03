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

/// Function state is set to 4 field elements or 192 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 4;
/// 3 elements of the state are reserved for rate.
pub const RATE_WIDTH: usize = 3;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 2;

/// One element (48-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = 1;

/// The number of rounds is set to 14 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 14;

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
    state[0] += mul_by_generator(&state[1]);
    state[1] += mul_by_generator(&state[0]);

    state[3] += mul_by_generator(&state[2]);
    state[2] += mul_by_generator(&state[3]);
    state.swap(2, 3);

    // PHT layer
    state[2] += state[0];
    state[3] += state[1];

    state[0] += state[2];
    state[1] += state[3];
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
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger384([
                    0x977efe83fc902a27,
                    0xc364ce2ec3d72c75,
                    0xf854d6a9ec7a6266,
                    0x4e29b2044d22d3b9,
                    0xce0809285647601a,
                    0x0cf54d2164d222e1,
                ])),
                Felt::new(BigInteger384([
                    0xc4713af3e73d95e8,
                    0x04a95bb850526c0f,
                    0xb90688c27b26ff9f,
                    0xcad8a9a2d5d267b0,
                    0x49469775f97c9774,
                    0x18db0d168af2616e,
                ])),
                Felt::new(BigInteger384([
                    0xc365f42bce627bad,
                    0xaf831f6aa541462e,
                    0x77f43759645824fd,
                    0x3f5f3358234b3c37,
                    0x9664e3ca489f5bc3,
                    0x0b6700c1ac89c1ed,
                ])),
                Felt::new(BigInteger384([
                    0x41394aa0d55d63ec,
                    0x7108963632453f08,
                    0xc35f0064818ed01a,
                    0x93deec86afa471a5,
                    0x711a4b0ddbc98f3b,
                    0x00580133b068fd6a,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd230948c57005945,
                    0xe79f24e9772189ee,
                    0x6cd56334fca738a3,
                    0xb682e04ad3ad86b0,
                    0x590810eed8814ad8,
                    0x02d8d3174d5e66a6,
                ])),
                Felt::new(BigInteger384([
                    0xca62b741f9b5db21,
                    0xccf389345c5d1887,
                    0xb66131047b38ffc5,
                    0x21da302798789890,
                    0xd789295de4dd4cf4,
                    0x0fa1eb1f42c5449a,
                ])),
                Felt::new(BigInteger384([
                    0x4d484df94b86fb9c,
                    0x4e5fe5538dccc797,
                    0xa4f752d9351a4005,
                    0xc5c0702fc81fb11e,
                    0x9250c2e6fe42a613,
                    0x167e0df1308a4887,
                ])),
                Felt::new(BigInteger384([
                    0x597ae666384cfb09,
                    0x4f8458268687919c,
                    0x2e7ea90ef3b01d8b,
                    0x4a286f641af8d060,
                    0xfa52845677f06f44,
                    0x04740e42017327ea,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6303525c26c6a4b4,
                    0x4b9c10ea37a67d4a,
                    0x7083523b987d9684,
                    0xaf4483465ffec6e5,
                    0x495e76bf13dabd8f,
                    0x0c5b93af6ef44f49,
                ])),
                Felt::new(BigInteger384([
                    0x393511ed0297e907,
                    0x6c9d25e845fda44c,
                    0xfe0c0211dc907b32,
                    0x54d9f9b3114a6c8c,
                    0x6ef418dd165c2261,
                    0x039ee499c43fb4dc,
                ])),
                Felt::new(BigInteger384([
                    0x307a8b7c138365d5,
                    0x28ab72f84cd04b90,
                    0xdb8549e263442d10,
                    0x67252ada702753e5,
                    0x4404b8f2e1ebf61a,
                    0x0b12bbe47f772186,
                ])),
                Felt::new(BigInteger384([
                    0xbe33b06e73188fdf,
                    0x05e976e4839514a0,
                    0x95db0d2634c3d393,
                    0x75a8a8f60d3f5930,
                    0x3bd96d8996f42e52,
                    0x1835c2d61da6fce8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x285a4a62cace1331,
                    0x6f40515e7420672f,
                    0xdf39eb40001c08bf,
                    0x76919b9ac3ed003e,
                    0x9a1681eaf6b253d5,
                    0x082f814d1d8133ca,
                ])),
                Felt::new(BigInteger384([
                    0x8f64077dc82ff042,
                    0xc5df0f1077013dfc,
                    0xc19c46a28592d2ec,
                    0x96664ad631c09c05,
                    0x0ed5eb4a22a251d9,
                    0x06804f0b03485e02,
                ])),
                Felt::new(BigInteger384([
                    0x7c81fb42f9174035,
                    0xf5f419dd53e7ad27,
                    0x9a3d287bfd421615,
                    0x1695b70a479990e5,
                    0x2be49f224032ba6d,
                    0x0f3029232a9fc762,
                ])),
                Felt::new(BigInteger384([
                    0x30ea60490be2862d,
                    0x6eaaefdfd71c039a,
                    0xab947993afb9d4e8,
                    0x7195fa636a9bf451,
                    0xe42d6d8a9ad98b4f,
                    0x0d967ec37b5b7818,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4d9a5402ecddda84,
                    0xc7f8ce9a5e16b9a2,
                    0x6fa957c23165820e,
                    0x17e55a709061ad1c,
                    0x9aa6ea2b699ac4e8,
                    0x0ae2e9b3344b97fd,
                ])),
                Felt::new(BigInteger384([
                    0xdd60459dbf654059,
                    0x03d9d56de3e07b1a,
                    0x3f0c6ed567cbf8e6,
                    0xba67491548dcb3b9,
                    0x8ba1bfb341cf05d4,
                    0x16c45f1014a468b7,
                ])),
                Felt::new(BigInteger384([
                    0x971ab309ac9e9a41,
                    0xf61d541b1b373d88,
                    0xde0954b09f3af51f,
                    0x9a3621066ef848b5,
                    0xf5024287ed77d766,
                    0x136ac542b7cf2091,
                ])),
                Felt::new(BigInteger384([
                    0xe9d28fa5d0773e6e,
                    0xdecf5104f0ad21c6,
                    0x6eedb59c94ef1637,
                    0xc360d1b56b86d0ed,
                    0x9cae4afdac058cde,
                    0x0c3e4e4c304b032f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x62e4b6aad419b2a5,
                    0x063228fd21c7f1fc,
                    0x02bc01dea6032e1a,
                    0x8ad648ebce21362c,
                    0x7613a4efba478c96,
                    0x09e8ddceacdee61e,
                ])),
                Felt::new(BigInteger384([
                    0xfb8c52c5cea4419b,
                    0xc69f4af9e04a92ef,
                    0x786f4ca14dd9f04e,
                    0xda8549a81e65435f,
                    0x2b919b13c71b914c,
                    0x14c94f663179f4f7,
                ])),
                Felt::new(BigInteger384([
                    0x017c7dbbf8bbe911,
                    0x684923a7f16587f3,
                    0xaeb93bc3d1f73254,
                    0x170b0cd31b991037,
                    0x51b519a69753485d,
                    0x144f58d9c68ceca2,
                ])),
                Felt::new(BigInteger384([
                    0x2cdfcd6770952212,
                    0x3cde884aec92e45a,
                    0xceb7336f42d4920c,
                    0x5cfcdc596d02e696,
                    0x872438d63e63c831,
                    0x0e684128061e257a,
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
                    0xec8dcfb387884ce9,
                    0xab62343b09125f11,
                    0x50b8fb4145ab3007,
                    0x35ace9d2c37a5b04,
                    0x44e00852e7f699b1,
                    0x03922f55c1970749,
                ])),
                Felt::new(BigInteger384([
                    0xf92b230cc9c6a629,
                    0x217eb90261f3af0a,
                    0x1b0808c563656422,
                    0xd5f254eacc8ea5aa,
                    0xd7a21915080e991e,
                    0x03ea9dfec6c492dd,
                ])),
                Felt::new(BigInteger384([
                    0xf4a5a1a95a4f5c79,
                    0xf8b5f8070b8cd05b,
                    0x90bc36ac82609ed4,
                    0xfbbcea8bfca45f28,
                    0x812c992d5969c6f0,
                    0x0f4ca0a3edf0d710,
                ])),
                Felt::new(BigInteger384([
                    0xa7267a74e6f845d8,
                    0xc6bfd339fc7f7f4d,
                    0x12dfe3f89e8bafc7,
                    0xb46665483df71046,
                    0x219955a750206397,
                    0x159ac442e11e1181,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4923beabb69b506f,
                    0x0c9326a86139b710,
                    0xdbea02bc0488013d,
                    0xa9751ecf3c0e38ad,
                    0x9e16c339ce19ada3,
                    0x04fcfe649c2bcf75,
                ])),
                Felt::new(BigInteger384([
                    0x3c83fd3b35f2f3df,
                    0x9b7fbaac954d3dcb,
                    0xc17bb6dd5c6684d2,
                    0xe471a2f93bc17709,
                    0x3329bfde647f2c8e,
                    0x0de0f7c6a18bd7b8,
                ])),
                Felt::new(BigInteger384([
                    0x6aaf1663ad76e17b,
                    0x7854b49acf4b5834,
                    0x5d8766f59f1aeef9,
                    0x929c7428af57a10c,
                    0x97ef82d0a68c2a99,
                    0x07170399e5b354cb,
                ])),
                Felt::new(BigInteger384([
                    0xbc113427d1c6ed13,
                    0x6b6d2381d30c53ad,
                    0xf87b15bb01fd20bf,
                    0x8b6e4c03aa50ad39,
                    0x5f44be1545e898ba,
                    0x17131bd5a8607aeb,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9aece39a1bc2e8c8,
                    0x0f1f316886da6586,
                    0xd56fcee0e4876dbc,
                    0x2e34bf9a5c53f445,
                    0xf919a8c384243252,
                    0x037f9d2a14816641,
                ])),
                Felt::new(BigInteger384([
                    0x6da82c02097724f9,
                    0xbf34cb13bae16379,
                    0x3bd080ce5612ac45,
                    0x13905e93347a5d95,
                    0x265fada4e1240273,
                    0x1987785b89cbd86b,
                ])),
                Felt::new(BigInteger384([
                    0x2d676fad0edfdd19,
                    0xea55ba723dd14211,
                    0xa07cdd73fdef43c5,
                    0xdf29dbb554081940,
                    0x84a3dce4b80b97a6,
                    0x168daa7df67c209e,
                ])),
                Felt::new(BigInteger384([
                    0xd166bfb9ecfc8b61,
                    0x1fe745320560fe20,
                    0xce23d07bc10d6170,
                    0x59c3a7cfcdb5deb5,
                    0x17021321e95891b2,
                    0x183f180caea8f001,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9f73206fe6e6f2fc,
                    0x071977b12a2ec0f4,
                    0xa7a21a04d774edea,
                    0x3e24e6eb027f8350,
                    0x5978cefff23849ef,
                    0x0641e1e8f3e3f683,
                ])),
                Felt::new(BigInteger384([
                    0x1f4ba3f001a9afbc,
                    0x76484dfb6de94a72,
                    0x07c4713cd06a9938,
                    0xaa0e8e3accace320,
                    0x7b71d079fc2422d3,
                    0x15a7d922da07747a,
                ])),
                Felt::new(BigInteger384([
                    0xf78bd6c606bc1d02,
                    0x1d578ff23846dc68,
                    0x6aac0cd9c05d657b,
                    0x340cbc4425a4d14b,
                    0xd166fccada1e3b92,
                    0x147867eb84dfa2e5,
                ])),
                Felt::new(BigInteger384([
                    0x299a15760f45461c,
                    0x38b7f5e3319ce508,
                    0xf300aaf2560af6bb,
                    0x3fd427423fe36c35,
                    0x371602a64d4594b7,
                    0x0eb1df073dd100e5,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6f62751c28f1503c,
                    0x3555d708a78b7c69,
                    0x77ec74f054ea27ad,
                    0xf9fdf54c7b63b2f8,
                    0x5cdf06d9425b7403,
                    0x0d61cfa4e39d65be,
                ])),
                Felt::new(BigInteger384([
                    0x14bb23237ab59882,
                    0xe45bfbb3d4c5bd4b,
                    0x898d1e76b6218a7d,
                    0x19f49b29438bc016,
                    0x686e4fa65d964acd,
                    0x06b256ada138ff78,
                ])),
                Felt::new(BigInteger384([
                    0x84c385b3fd1dee11,
                    0xa3ad5b5c69d3a6e2,
                    0xf5dc3228fb7f6c41,
                    0xe35240bb9a92b46f,
                    0x391c6204473ddaa3,
                    0x12f8be20abbf23e8,
                ])),
                Felt::new(BigInteger384([
                    0xcdefe021a3746293,
                    0x6aff8800c7b7fa12,
                    0x5048843e9d506e60,
                    0xa13b197bb2cd009c,
                    0xed1dc44c42e10d26,
                    0x0d4c87cfd4c1c0fc,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7e79deede94f4e3b,
                    0xaeeed5c4ab0b8d29,
                    0x2c0b5062cc2bff77,
                    0x4094975aaac76495,
                    0xa40747d92cc31ee9,
                    0x1815e0ebf0dda5c5,
                ])),
                Felt::new(BigInteger384([
                    0xadb7408e66587c1f,
                    0x697258f19105d97c,
                    0x4c3ee36b494139a7,
                    0xade59fd0021171ad,
                    0xb96991f758bac50d,
                    0x01d87821e4b47b52,
                ])),
                Felt::new(BigInteger384([
                    0x093f64903d2539b6,
                    0x9e25c763343ad61e,
                    0x9a47d4ab3c47c5c4,
                    0xdf7ab1e4bdbeb7ee,
                    0x17014559bf651e97,
                    0x079471786c01d33f,
                ])),
                Felt::new(BigInteger384([
                    0xebe124ffc9c97472,
                    0x235bd04769c0a966,
                    0xbee0bc35849d9f82,
                    0x8ae00b7a067521a2,
                    0x7c612549a97b9e0e,
                    0x1617afc562103d80,
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
