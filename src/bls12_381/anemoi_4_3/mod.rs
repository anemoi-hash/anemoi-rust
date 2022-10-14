use super::{mul_by_generator, sbox, BigInteger384, Felt};
use crate::{Jive, Sponge};
use ark_ff::{Field, One, Zero};
use unroll::unroll_for_loops;

/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// MDS matrix for Anemoi
mod mds;
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

/// The number of rounds is set to 12 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 12;

// HELPER FUNCTIONS
// ================================================================================================

#[inline(always)]
/// Applies exponentiation of the current hash
/// state elements with the Anemoi S-Box.
pub(crate) fn apply_sbox(state: &mut [Felt; STATE_WIDTH]) {
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

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    state[0] += mul_by_generator(&state[1]);
    state[1] += mul_by_generator(&state[0]);

    state[3] += mul_by_generator(&state[2]);
    state[2] += mul_by_generator(&state[3]);
    state.swap(2, 3);
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

    apply_mds(state)
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

    apply_mds(state);
    apply_sbox(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_naive_mds(state: &mut [Felt; STATE_WIDTH]) {
        let x: [Felt; NUM_COLUMNS] = state[..NUM_COLUMNS].try_into().unwrap();
        let mut y: [Felt; NUM_COLUMNS] = [Felt::zero(); NUM_COLUMNS];
        y[0..NUM_COLUMNS - 1].copy_from_slice(&state[NUM_COLUMNS + 1..]);
        y[NUM_COLUMNS - 1] = state[NUM_COLUMNS];

        let mut result = [Felt::zero(); STATE_WIDTH];
        for (i, r) in result.iter_mut().enumerate().take(NUM_COLUMNS) {
            for (j, s) in x.into_iter().enumerate().take(NUM_COLUMNS) {
                *r += s * mds::MDS[i * NUM_COLUMNS + j];
            }
        }
        for (i, r) in result.iter_mut().enumerate().skip(NUM_COLUMNS) {
            for (j, s) in y.into_iter().enumerate() {
                *r += s * mds::MDS[(i - NUM_COLUMNS) * NUM_COLUMNS + j];
            }
        }

        state.copy_from_slice(&result);
    }

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/vesselinux/anemoi-hash/
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
            apply_sbox(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }

    #[test]
    fn test_mds() {
        // Generated from https://github.com/vesselinux/anemoi-hash/
        let mut input = [
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger384([
                    0x472b65e69c9d50f7,
                    0x66fd4247d8daef1a,
                    0xf62e6a7fea6ea3bf,
                    0xb786e732918adb4c,
                    0xfdf84c7b49fb83b3,
                    0x13b75a1e71551217,
                ])),
                Felt::new(BigInteger384([
                    0x0bad515fea8d3b37,
                    0x2b700df149cc0168,
                    0x049fd4f876e6dc95,
                    0x506cbc1da2245804,
                    0x01dd7983dd73cf98,
                    0x142ae61e0caa09c0,
                ])),
                Felt::new(BigInteger384([
                    0x5d7cd372d14e94f0,
                    0x5a7027080312d9fe,
                    0x96ae154c86a9befb,
                    0x785a1a5af46a86cf,
                    0x1a5fe9cfa0cd1024,
                    0x0e696d18f3293fb6,
                ])),
                Felt::new(BigInteger384([
                    0x51f3e0a3618b1a32,
                    0xf561ddafc3aff821,
                    0xc080705130795c17,
                    0x341ea020d7baf201,
                    0xd34bfe82ce333b97,
                    0x0e2a65f8a921680d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xdab238fad7f26853,
                    0xdf5efbe739d7e2b2,
                    0x499c6f4e1dd132dd,
                    0xd961a3f2cf9ac667,
                    0x109e6ba432f6144a,
                    0x0159576889dcb22d,
                ])),
                Felt::new(BigInteger384([
                    0x648db6085cfc044b,
                    0xaafc7afd7554fb49,
                    0x03b42efeec756ff6,
                    0x5e81c34336445669,
                    0x6fe1135a00c55ae2,
                    0x05b050240c5b057a,
                ])),
                Felt::new(BigInteger384([
                    0xa88a360867319098,
                    0xae988542b734c57c,
                    0x1cb067ad14feed2a,
                    0xbdb753c208848892,
                    0xc3dc100bf7a4c1e8,
                    0x162d3a0f3d24d400,
                ])),
                Felt::new(BigInteger384([
                    0xb13e4574a579380d,
                    0xd551aac57d7655c3,
                    0x7b21dd2497c2518c,
                    0x59ce537cde426d7a,
                    0xf56f145b95a8d6df,
                    0x03f62d4b5bf7c067,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0ed704259ac4670c,
                    0xdec50ca864c31524,
                    0x4186c8f187fd90b7,
                    0x360f2c346af75a70,
                    0x853636c4599b1489,
                    0x108e85a75e309191,
                ])),
                Felt::new(BigInteger384([
                    0x45415bf9b8eafefb,
                    0x140698508ec925e9,
                    0x4469003967e38a37,
                    0x0ecb0619b17a310a,
                    0x1543311dfb12e6e5,
                    0x10f4ddcdb83a50f4,
                ])),
                Felt::new(BigInteger384([
                    0x03e292fb99fe78bc,
                    0xd59adc6ca9a07d20,
                    0xd93611e0b5cbf364,
                    0xf9c4bed6b2f9ed66,
                    0xc02198d24bc9d780,
                    0x12496969ba69baca,
                ])),
                Felt::new(BigInteger384([
                    0xde3418dba15fd8e6,
                    0x9ef63b9c273ba39d,
                    0xd9f579b09ac1560d,
                    0xf2db48fd6fc9a8c8,
                    0x9597042f6ff5b7e6,
                    0x11a0160e108f06fc,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3bb2a7ca6b58b2a3,
                    0x43e8e74746e10a3d,
                    0xdf8a2108880ca586,
                    0x7f31548683d262f8,
                    0x350597a147a9f780,
                    0x0005ad6783805c39,
                ])),
                Felt::new(BigInteger384([
                    0x06826233446d6415,
                    0x921597a119b9e6fb,
                    0xf3cd26e6f0499fde,
                    0x4da510d83c854d98,
                    0x12ae06bd2b46ed6d,
                    0x071b4bad8deda893,
                ])),
                Felt::new(BigInteger384([
                    0x4370aa66045518ad,
                    0xb5fd2c4feb30f610,
                    0xfaae489557123a46,
                    0x162d7dc35d2ef379,
                    0x4030ff7f5ebe8cdc,
                    0x12d263400e14f1a1,
                ])),
                Felt::new(BigInteger384([
                    0x043ebf85c732a89d,
                    0xb2f940d987babb23,
                    0x3c224ea10a66b0b5,
                    0x26d2465d50634256,
                    0x19111486ee87d101,
                    0x154ead767cae96e9,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xecbcc5bfe9fb3c58,
                    0x8bc49975fd98fe2d,
                    0x389a5f28dc563597,
                    0x4d7b62bd54175bde,
                    0xdc5e6ea101b6c232,
                    0x0d8730e7aaec4346,
                ])),
                Felt::new(BigInteger384([
                    0x6fa220dbf7eccf88,
                    0xa6c59cf4f8c093ae,
                    0x5732d2214815d94e,
                    0xf4f62c83a5e9d482,
                    0xa0cb2180d532c07d,
                    0x1892847f2828f258,
                ])),
                Felt::new(BigInteger384([
                    0x8eb939ed3701126d,
                    0xd79d231f8cae88f4,
                    0x7d2821c445a063e3,
                    0xa8da3d8590925449,
                    0x42c4a90c39ecaf24,
                    0x173ebef9c94ad861,
                ])),
                Felt::new(BigInteger384([
                    0x070646e34ccbd1c2,
                    0x26a1be2ee3df756c,
                    0x20302d1982a7ea5c,
                    0x0a691ece06a8d77b,
                    0x42fb5a1aab228a3c,
                    0x07f728b54b45eeb8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe435532f272c58b2,
                    0x9c136f95cb585db0,
                    0xabc229d49e49f705,
                    0x45bfb3d7bac9f7ec,
                    0x7fccc95f3a45dbf3,
                    0x17c246868961d335,
                ])),
                Felt::new(BigInteger384([
                    0x09eda1fa989d7c45,
                    0xa26eefa8e0426a5e,
                    0xb16085cb47cf0a20,
                    0xc3671c87f1c3a6bd,
                    0x2e7fccb7e8989dc9,
                    0x0299697869b400e2,
                ])),
                Felt::new(BigInteger384([
                    0x10764089743fc178,
                    0x98e85b765a1f53f3,
                    0x95a46022dd9ce6ab,
                    0x4692f14963d9ea8f,
                    0x2a7cd7926a89471e,
                    0x080d58c0131d673d,
                ])),
                Felt::new(BigInteger384([
                    0x0b9b55a2208b57bb,
                    0xbfebf4ff59acc759,
                    0x25fc6b15011438e2,
                    0xc0ccf62cd48ef33c,
                    0x5b6fa8dc80ab9282,
                    0x03cfaa6bcfd7bb78,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 4],
            [
                Felt::new(BigInteger384([
                    0xee1d00000009aaa1,
                    0x86840025e97c0007,
                    0x4f7823c40df41de8,
                    0x9e7c71f069ece051,
                    0x7dde005a606d6b99,
                    0x0de0f8777c82e085,
                ])),
                Felt::new(BigInteger384([
                    0x984400000016aa94,
                    0xda500058e5b00012,
                    0x97080d3e78fe9e67,
                    0x504ff0af50a70528,
                    0x0ca773961fe616c9,
                    0x17b73dc8ba06bf04,
                ])),
                Felt::new(BigInteger384([
                    0xee1d00000009aaa1,
                    0x86840025e97c0007,
                    0x4f7823c40df41de8,
                    0x9e7c71f069ece051,
                    0x7dde005a606d6b99,
                    0x0de0f8777c82e085,
                ])),
                Felt::new(BigInteger384([
                    0x984400000016aa94,
                    0xda500058e5b00012,
                    0x97080d3e78fe9e67,
                    0x504ff0af50a70528,
                    0x0ca773961fe616c9,
                    0x17b73dc8ba06bf04,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger384([
                    0xee1d00000009aaa1,
                    0x86840025e97c0007,
                    0x4f7823c40df41de8,
                    0x9e7c71f069ece051,
                    0x7dde005a606d6b99,
                    0x0de0f8777c82e085,
                ])),
                Felt::new(BigInteger384([
                    0x984400000016aa94,
                    0xda500058e5b00012,
                    0x97080d3e78fe9e67,
                    0x504ff0af50a70528,
                    0x0ca773961fe616c9,
                    0x17b73dc8ba06bf04,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xee1d00000009aaa1,
                    0x86840025e97c0007,
                    0x4f7823c40df41de8,
                    0x9e7c71f069ece051,
                    0x7dde005a606d6b99,
                    0x0de0f8777c82e085,
                ])),
                Felt::new(BigInteger384([
                    0x984400000016aa94,
                    0xda500058e5b00012,
                    0x97080d3e78fe9e67,
                    0x504ff0af50a70528,
                    0x0ca773961fe616c9,
                    0x17b73dc8ba06bf04,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger384([
                    0xea8808a671b8720f,
                    0x80855e2d09caf1ea,
                    0x310c6f2eeada70a1,
                    0x8f71c863eec965d6,
                    0x6b7bf0167e4bc935,
                    0x080b028617a95863,
                ])),
                Felt::new(BigInteger384([
                    0x26be62accdfe74aa,
                    0x0dceca4cac0de53e,
                    0xff87e0b555eac7b4,
                    0x0ad901608c3210f0,
                    0x8db9b1fa96bfb52c,
                    0x0a3fd940027cd3ec,
                ])),
                Felt::new(BigInteger384([
                    0x52ee878904289967,
                    0x8b962bc11881ac1e,
                    0x86abc849471be3ea,
                    0xc05b8951cd0aece1,
                    0xbcf02a6bcc81af08,
                    0x10fc2e4055f400df,
                ])),
                Felt::new(BigInteger384([
                    0x495ae284d9a01d13,
                    0x52f07e8b82c2323b,
                    0x3cd4d33e1e3090ac,
                    0x9499e1799afb4dd3,
                    0x492496f0f684c15e,
                    0x1660b7af65915adb,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xa3cda50b91ea70e9,
                    0x3557f1e22481d945,
                    0x5104cd4bf6bc12cb,
                    0x96652a793c237339,
                    0xf06092583480ca0f,
                    0x0cb9f7b0a292bd21,
                ])),
                Felt::new(BigInteger384([
                    0xf22a001f80d13b72,
                    0xf7005ec30d04add4,
                    0x3e8cf6f5e33c9f68,
                    0x26d4ccb0bb062a1c,
                    0x05869054267b422a,
                    0x05232d9b18009924,
                ])),
                Felt::new(BigInteger384([
                    0x4853b18573dcae92,
                    0x13d6b54c3a8be0bd,
                    0x4d51d9ddcb0f35be,
                    0x70c5af7bfbc66bdf,
                    0x320b8cbd41a6add9,
                    0x164f8f7f9cc181cf,
                ])),
                Felt::new(BigInteger384([
                    0xc53399134eeb9866,
                    0x98edefddc9a486f7,
                    0xe8f27626bdbb6c5e,
                    0xd6541bb019073ad1,
                    0x91bbda19f45ac3ec,
                    0x0eca353a03a80a6a,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xdf5abc190c9aba57,
                    0xe8263d4ad10160f6,
                    0x6327f6c36113af01,
                    0xef2dece2da66a9c5,
                    0x64a0f14a0c75357b,
                    0x18772f5895254cdf,
                ])),
                Felt::new(BigInteger384([
                    0x8ff8d42bd2211e53,
                    0xa6fb12e8ce23e7d7,
                    0x3c57487e3ca8fbf2,
                    0x243848d57f3d5f16,
                    0x484dc4458d65f82e,
                    0x0de118aa6f851d7e,
                ])),
                Felt::new(BigInteger384([
                    0x71fb3ed2d55d7508,
                    0x0cd3f47817d49dde,
                    0xbdfff83018f7508f,
                    0x1d762fa0eeb35e17,
                    0x7fa2e66780f20d3a,
                    0x0230c50d1262af5d,
                ])),
                Felt::new(BigInteger384([
                    0xe7d910a144b962cc,
                    0xef42c55cd949b8dc,
                    0x55360240e7ba9482,
                    0x34b11e189060a996,
                    0xbf6765a14dadf1f5,
                    0x16aaf383df2f1985,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x48b76c30f4337acd,
                    0x681416897a54d833,
                    0xc7246ed6689fe543,
                    0x1a7b7636fcdcfe2a,
                    0x5a61a51b9e37d25b,
                    0x0e3c44c29f5bad5f,
                ])),
                Felt::new(BigInteger384([
                    0xddf23a952cd4af04,
                    0x4391c4b55d0f9761,
                    0x1ae531f2cad87441,
                    0x1e24b1c142ba372f,
                    0x7c55a93e246ae54c,
                    0x0992c34893251cb7,
                ])),
                Felt::new(BigInteger384([
                    0x17221451cfdd84a1,
                    0xe19b997bfb74a744,
                    0x631d3a89cb2938fa,
                    0x8a3eaada23b703cb,
                    0x033bc419256d910a,
                    0x06f1502225d8acf7,
                ])),
                Felt::new(BigInteger384([
                    0xb7b5d309a4107744,
                    0x5a885f4930c64498,
                    0x59b7eb07f6b3b618,
                    0xc63387f2b117e851,
                    0xfb8cdffb664e0219,
                    0x06b3f19a204664f4,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x58030777d9d58612,
                    0x9bf7d3628c72258b,
                    0x189e5e297f1ffbec,
                    0x6e7924bab8e0df64,
                    0x87bd62362584e97f,
                    0x0aaa1611883e5ac3,
                ])),
                Felt::new(BigInteger384([
                    0x65a92fcbab983101,
                    0xc00943bb6050dec5,
                    0x213ebbd34fa4db03,
                    0x6d712a742426808b,
                    0x652a3e36dcf0e6a5,
                    0x13e59eb7ff25c145,
                ])),
                Felt::new(BigInteger384([
                    0xb07ababdbacea146,
                    0x988404709a948755,
                    0x4c1ecb602086c5db,
                    0x932f02cf40c35a8f,
                    0x324d5cc698648ed6,
                    0x027282d46adbd246,
                ])),
                Felt::new(BigInteger384([
                    0x35afaf68ac9eaa4e,
                    0xe9f92c02108397a0,
                    0xae34e5e38ffcf976,
                    0x6ac0f79f1e93f6a8,
                    0x5c43bae3276a1ffa,
                    0x0222b2b865829653,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3e1197245867a691,
                    0xc2454ee8da89326d,
                    0xa75262ca37371522,
                    0x6816a162aacc32a8,
                    0x91b0bb18c82b6aaf,
                    0x02f4078d2349ee5f,
                ])),
                Felt::new(BigInteger384([
                    0x8610d043496cc967,
                    0x26f98d7a9554cf38,
                    0x00054b5fb63d3466,
                    0x93945f4d475c0c0f,
                    0x51e142e978ef7328,
                    0x08817892b047dda1,
                ])),
                Felt::new(BigInteger384([
                    0x2c87d6b5090adaab,
                    0xf1bcabec0deb6f3f,
                    0x51452b5abc4e0639,
                    0x4df2d8bf9c42c85b,
                    0xb069580155be20bf,
                    0x13ea5bebf61289f2,
                ])),
                Felt::new(BigInteger384([
                    0xaf86edf38655cc23,
                    0x5db5b34fc4a23271,
                    0xd0fde4375f87fcfb,
                    0x7e015743a8da6886,
                    0x4033dfded2b9dbc5,
                    0x15e0feadc5c29488,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            apply_mds(i);
        }
        for i in input2.iter_mut() {
            apply_naive_mds(i);
        }

        for (index, (&i_1, i_2)) in input.iter().zip(input2).enumerate() {
            assert_eq!(output[index], i_1);
            assert_eq!(output[index], i_2);
        }
    }
}
