//! Implementation of the Anemoi permutation

use super::{mul_by_generator, sbox, BigInteger256, Felt};
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

/// Function state is set to 4 field elements or 128 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 4;
/// 3 elements of the state are reserved for rate.
pub const RATE_WIDTH: usize = 3;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 2;

/// One element (32-bytes) is returned as digest.
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
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x3898d6276e250a1a,
                    0x49ef67ce2abb8cd5,
                    0x67d3f7ac7aa535b1,
                    0x3be695163c5779ea,
                ])),
                Felt::new(BigInteger256([
                    0x07a443be9662fb4b,
                    0x781fadb683f44719,
                    0x2e93c0d94026aa4d,
                    0x0f9f9eb4964f5978,
                ])),
                Felt::new(BigInteger256([
                    0xe44e1c928faaad5b,
                    0x3e274fda062783e7,
                    0x6d4446c6a63f8126,
                    0x3f2595c15666d614,
                ])),
                Felt::new(BigInteger256([
                    0x1784c93382b00592,
                    0x860d15ad49ed3feb,
                    0x2c029d26007924bb,
                    0x1e24d3509bc5cebb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x40c257f79e66756f,
                    0x2740d9e2fad77116,
                    0x355155c0270ee015,
                    0x10666ec0fcb9f674,
                ])),
                Felt::new(BigInteger256([
                    0x862688a348b98cff,
                    0x5fccf768cc8241ec,
                    0x31be221dca2c996f,
                    0x3f417de5e31b44b6,
                ])),
                Felt::new(BigInteger256([
                    0x14015c2159cdc913,
                    0xbf61fba45f2ef946,
                    0x33135d0834e38671,
                    0x390c6566d5df6fad,
                ])),
                Felt::new(BigInteger256([
                    0x12224ed4d08c688d,
                    0xf228c8646f4936c1,
                    0x54e08e9821e0c1b0,
                    0x1595bbe195f2bf1c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4c9aa1d6b29ed549,
                    0xdb1b457283caa7ec,
                    0x64ebbc065f9b8df4,
                    0x0aa53d12795852d8,
                ])),
                Felt::new(BigInteger256([
                    0x4cf671ce385a018a,
                    0xacf8543356f9b541,
                    0xcb2b96fd651396fe,
                    0x2e26a10a41c5dacb,
                ])),
                Felt::new(BigInteger256([
                    0x645d9f95a9c56b1c,
                    0x022282bd008db7e1,
                    0xea2c55a1c02e4d93,
                    0x25eec9d245da0590,
                ])),
                Felt::new(BigInteger256([
                    0xb6cc96c5ce05aaef,
                    0x2118bbe41cab4473,
                    0x893c357f8887b564,
                    0x0f540ce8c8f4842a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x67018c5df6519203,
                    0xb9d4d7115150edfc,
                    0xfd53563d708296f4,
                    0x1b26c794741c0bfb,
                ])),
                Felt::new(BigInteger256([
                    0xfc62dfebfcb28f75,
                    0xaf3a0bb289ab9223,
                    0x82bb875fe1cae254,
                    0x384e091e28a6e58d,
                ])),
                Felt::new(BigInteger256([
                    0xd0fff7bf031a2ec9,
                    0xd5375bd9b2406182,
                    0xa0130e277ba41f2b,
                    0x0d019209e0f0fffe,
                ])),
                Felt::new(BigInteger256([
                    0xa445297a25803869,
                    0xb8b3a1ba5c67d015,
                    0x42760d37afb9b3f4,
                    0x34d4abee9150e48e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc49c6c9b2eab6c16,
                    0x2b14ddb75b3bbf69,
                    0x47933ee5da4bb5da,
                    0x38c2a8f5a624cde6,
                ])),
                Felt::new(BigInteger256([
                    0x3bff0cb9a49b6b75,
                    0xe32bbc6d129fb9b4,
                    0x210f9051874dcaf8,
                    0x114e00bf71772669,
                ])),
                Felt::new(BigInteger256([
                    0x08f372532d9f69c3,
                    0x4a425194827fe44c,
                    0x098d0028f362ce6a,
                    0x059ca7adb0f75bf6,
                ])),
                Felt::new(BigInteger256([
                    0xb9687d81a06577fe,
                    0x8bb1e9fd2c90903f,
                    0xe29c8a6906f72039,
                    0x1401c0cd3c01bfc9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x81c2c3c90a71cb0b,
                    0xf653c0858343a8e5,
                    0xd77e9e8ac1d45b02,
                    0x3069aa4adeecdfd2,
                ])),
                Felt::new(BigInteger256([
                    0xdfc550f74beb99d5,
                    0x5503dd57c62d70d4,
                    0x71c0e72f4626b9c7,
                    0x20dde63c8d6510ea,
                ])),
                Felt::new(BigInteger256([
                    0x60efa57c8292993f,
                    0x2e828c75d5d147ff,
                    0xbb5fc2e37fb6ed48,
                    0x3daecf10d4987081,
                ])),
                Felt::new(BigInteger256([
                    0x08b53593f3a2825a,
                    0x8227b9a271486308,
                    0x56e41268cc3a4b5a,
                    0x082feacb1be7237b,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0x123bd95299999999,
                    0xb83c0a9bfa40677b,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::new(BigInteger256([
                    0x123bd95299999999,
                    0xb83c0a9bfa40677b,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xa942fcc6b6fc9019,
                    0xe30fee4001297b6b,
                    0xc81bbbb990652d5f,
                    0x151360a4ca1bade9,
                ])),
                Felt::new(BigInteger256([
                    0xa942fcc6b6fc9019,
                    0xe30fee4001297b6b,
                    0xc81bbbb990652d5f,
                    0x151360a4ca1bade9,
                ])),
                Felt::new(BigInteger256([
                    0xc1f5f40ca5f0c922,
                    0x8ed096798f9a2565,
                    0x343bf2efffdc45e4,
                    0x136f8d515fc00212,
                ])),
                Felt::new(BigInteger256([
                    0xc1f5f40ca5f0c922,
                    0x8ed096798f9a2565,
                    0x343bf2efffdc45e4,
                    0x136f8d515fc00212,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2daf6d68ef718427,
                    0x5061c937d5b3a764,
                    0xf8541bb755f5c730,
                    0x24e19c62db751b35,
                ])),
                Felt::new(BigInteger256([
                    0x2daf6d68ef718427,
                    0x5061c937d5b3a764,
                    0xf8541bb755f5c730,
                    0x24e19c62db751b35,
                ])),
                Felt::new(BigInteger256([
                    0xffc6f91ea63fca57,
                    0x291c6ce7a87d6cc6,
                    0x0dde5b4ffa4ac0ea,
                    0x0d38cb1a91ae29c0,
                ])),
                Felt::new(BigInteger256([
                    0xffc6f91ea63fca57,
                    0x291c6ce7a87d6cc6,
                    0x0dde5b4ffa4ac0ea,
                    0x0d38cb1a91ae29c0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xeb95ce3a99999981,
                    0x819db2fb145092b5,
                    0xccccccccccccccc9,
                    0x0ccccccccccccccc,
                ])),
                Felt::new(BigInteger256([
                    0xeb95ce3a99999981,
                    0x819db2fb145092b5,
                    0xccccccccccccccc9,
                    0x0ccccccccccccccc,
                ])),
                Felt::new(BigInteger256([
                    0x311bac8400000004,
                    0x891a63f02652a376,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
                Felt::new(BigInteger256([
                    0x311bac8400000004,
                    0x891a63f02652a376,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfbcfdaf527dfeed2,
                    0x12c843702ae02f05,
                    0xe5677fac6c3e397c,
                    0x2f4922ff09d3c7d5,
                ])),
                Felt::new(BigInteger256([
                    0x1c332a154a53782f,
                    0x76e951a15bc540c6,
                    0xe3208d13688477c4,
                    0x01039cd670bd7a84,
                ])),
                Felt::new(BigInteger256([
                    0x665b3c8de3e8dc02,
                    0x10bfb169f067047e,
                    0x147ba05c7a24622a,
                    0x07bebec7c6acec01,
                ])),
                Felt::new(BigInteger256([
                    0x9b2b46fdb836640a,
                    0xe16d7db5694a8609,
                    0x26bd067ea71cd626,
                    0x27eb560e7d23b91e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfe823e540178f6a7,
                    0xe6df56615b3674c8,
                    0x2faaf4cee87de130,
                    0x07bbe3bafe3f518a,
                ])),
                Felt::new(BigInteger256([
                    0x739c338c54d2fe55,
                    0xe8b0cec326283d83,
                    0x7eecb4ead08ef622,
                    0x338db6b9cd75c264,
                ])),
                Felt::new(BigInteger256([
                    0x8babb83eb3ac2782,
                    0x01032f65f0fe68b0,
                    0x35e6c8e7de51894b,
                    0x04b49d8d3bf5e9a5,
                ])),
                Felt::new(BigInteger256([
                    0x123e14bbf16a6436,
                    0x0350e75a5b1fa5b7,
                    0x904262a2e247ada6,
                    0x0bba51838d4edf4c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x81e63de4667216aa,
                    0xee6b73405c02e799,
                    0xaafe47b932e8952e,
                    0x228a1ccc791aa519,
                ])),
                Felt::new(BigInteger256([
                    0x5f94679e3608dce2,
                    0x4bf6cbfecd8bfcb0,
                    0x64b8df309a21eef7,
                    0x2fac1e1a56940682,
                ])),
                Felt::new(BigInteger256([
                    0xdf04602f4d49e2df,
                    0x40b4c3ffd1461212,
                    0xeb6d8fa65f3dbc6b,
                    0x36c0df7c4fbd4b7c,
                ])),
                Felt::new(BigInteger256([
                    0x9af53e39ce820e12,
                    0xa8aec40f79cae035,
                    0x44e1804e7f9b4a09,
                    0x18c56bf644485615,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04b00dd7113d1692,
                    0x7e02c3e3e4d4414e,
                    0x51d2ed025181aac7,
                    0x001e0e37b0bff1cb,
                ])),
                Felt::new(BigInteger256([
                    0x9e440667ca83b7c8,
                    0x674e17467df36f74,
                    0x1290ae479baa2741,
                    0x2be17968d5f0ac38,
                ])),
                Felt::new(BigInteger256([
                    0x86de9b30a6b7c042,
                    0x9a49e57c8ca0795e,
                    0xfd351d1a596615b0,
                    0x0cec52d8c421544a,
                ])),
                Felt::new(BigInteger256([
                    0xceb2af5bb19aa680,
                    0xb06c0ecf52c3d4fe,
                    0x49648df58c7e25c4,
                    0x19a1bff73ec23d04,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4a8dff05802409a7,
                    0x4f2f9311d37ade68,
                    0x95f54ed50b6e60da,
                    0x3842ffa543bb23a9,
                ])),
                Felt::new(BigInteger256([
                    0x0e129015807fe8db,
                    0xc5a9ae5ccca44e50,
                    0x915142c1b3f5b5c7,
                    0x0f764f19fb96ad90,
                ])),
                Felt::new(BigInteger256([
                    0x0423a6c005d2adb3,
                    0x573f72484253c322,
                    0x916e42da94cc487d,
                    0x0e71530d206eff2a,
                ])),
                Felt::new(BigInteger256([
                    0xe4d80d7bda046e62,
                    0x00b2ce60aa2da04e,
                    0xe363adb754a49307,
                    0x00acb620871a247a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x14971af392933ad5,
                    0x594a30d1bae47478,
                    0x4ed9b344b0fb7e55,
                    0x37f721f534104995,
                ])),
                Felt::new(BigInteger256([
                    0x89f30bd6ede1cffb,
                    0xb4f780c93f24381a,
                    0xbdc6c5cff8dfd85b,
                    0x1aa5ec6837db1d17,
                ])),
                Felt::new(BigInteger256([
                    0xcef78f2c31d44a62,
                    0x510b37dbc97cdcee,
                    0x9e3a8ab9cc8f1926,
                    0x006c4197590ff596,
                ])),
                Felt::new(BigInteger256([
                    0x635c0f809b89d105,
                    0xcb09bea77aa8a9f8,
                    0x5b5a6ceb8ef90767,
                    0x1866b1450e2d488e,
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
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x7118de32f35694c9,
                    0x1a39653b5c4107db,
                    0x4622739ab7cac262,
                    0x2d3ba771b01c9afe,
                ])),
                Felt::new(BigInteger256([
                    0xd9b75e4be617add4,
                    0x2fc25dca6620d36f,
                    0x8d281097ca2d2e30,
                    0x02c9da255c6d792d,
                ])),
                Felt::new(BigInteger256([
                    0xd1a60c7e90a3e1da,
                    0x180c15bc4bf46917,
                    0xbf77985b8ab9d217,
                    0x00bd68ba86c5d34c,
                ])),
                Felt::new(BigInteger256([
                    0x8a39382514942b4e,
                    0xde22c44bfc9b9516,
                    0x5d0b0304c8075c8b,
                    0x2e8c683be67bc28e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x447ee8989bf4a547,
                    0x2e7236c418d678cb,
                    0x3667de81336a2441,
                    0x0ac6e74493788af3,
                ])),
                Felt::new(BigInteger256([
                    0x19527076a61dd65a,
                    0x4d3a70d7f7c23b3d,
                    0x85e1950bab65d0ed,
                    0x0cd5474c1d49a538,
                ])),
                Felt::new(BigInteger256([
                    0x5941aa2f6f94e46c,
                    0xc4d976124640ffef,
                    0x282109fb2abe2b54,
                    0x017d1e235ebdaa52,
                ])),
                Felt::new(BigInteger256([
                    0xcb270274a5723b11,
                    0xc99f0a3b39afcb8d,
                    0xda07466a3da7ffcc,
                    0x08a58de357309207,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb374cc4d37c643d5,
                    0x062c1cde40d860ac,
                    0x48dea503a6b0ded1,
                    0x140fbc585715df73,
                ])),
                Felt::new(BigInteger256([
                    0xa3bf12573b6d5c61,
                    0x8bc640761628472d,
                    0x8e53b8d5e8a9982b,
                    0x0036fb5d0f6f9bd5,
                ])),
                Felt::new(BigInteger256([
                    0xd392243aa8229cdf,
                    0x5f15bae0e961c30a,
                    0x28c662eb549b23f3,
                    0x1861b00320edc757,
                ])),
                Felt::new(BigInteger256([
                    0x9c90888a792036f2,
                    0x51bdfb743ff9caaf,
                    0xca5e9bd933130929,
                    0x25089c3af5ba7731,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb7525e84e9bb9934,
                    0x9d79f12692770d3b,
                    0x6b94242e32ef3075,
                    0x3efbc099828b29c4,
                ])),
                Felt::new(BigInteger256([
                    0xda29f9b1decc9506,
                    0x0df3450087af7f70,
                    0xf13c1aa7622bed3b,
                    0x3a0456f9a9b90f19,
                ])),
                Felt::new(BigInteger256([
                    0x1bcea61482efa161,
                    0x511d167f8c7beb53,
                    0xe9dbe62f44341759,
                    0x0103a9d2c565c449,
                ])),
                Felt::new(BigInteger256([
                    0x1b947ae8da147c63,
                    0x2228cd300130fe47,
                    0xaf1d8693ac67f6b2,
                    0x2da57590e4664496,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x611c5b08b891c43f,
                    0x6bdc16a62da0f7ee,
                    0xf887447cf726fad3,
                    0x096bc7d49c243335,
                ])),
                Felt::new(BigInteger256([
                    0x467ece9b797d902e,
                    0x2694a07c25ead118,
                    0xe4f076a23a0e3f08,
                    0x3de15b3dc016c250,
                ])),
                Felt::new(BigInteger256([
                    0x223320f1c764bb5f,
                    0xe473b1fb440031c7,
                    0x388a667ebcfddd34,
                    0x13f6b32d76dc3bd2,
                ])),
                Felt::new(BigInteger256([
                    0xca976d520dafdcd2,
                    0x72abb2a0c177b054,
                    0x3f5b6ccf099195f5,
                    0x1a2a3e854e627e2d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb15b79d7285488b2,
                    0x662a83b6740e7583,
                    0xfaa201a1e1f448f5,
                    0x09e152dd80e5f153,
                ])),
                Felt::new(BigInteger256([
                    0x3f72b42691d82e37,
                    0x7dbd1059df82bc1f,
                    0xa12812fca99b600c,
                    0x0e6cf0792eab6bbb,
                ])),
                Felt::new(BigInteger256([
                    0x654baba922b1e90a,
                    0xd62f0a24c224f40d,
                    0x8a2aecdeefdf12a9,
                    0x0cc3cb3dcd4dccf0,
                ])),
                Felt::new(BigInteger256([
                    0x5a73a76eb2bd0656,
                    0x57914607146508ed,
                    0x9b03b738c02db355,
                    0x0b4400dae8701b6c,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 4],
            [
                Felt::new(BigInteger256([
                    0x65a0e008ffffffe9,
                    0xeba8415b23a4d418,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x99ed0724ffffff85,
                    0x88147ee76592dd8d,
                    0xffffffffffffffef,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x65a0e008ffffffe9,
                    0xeba8415b23a4d418,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x99ed0724ffffff85,
                    0x88147ee76592dd8d,
                    0xffffffffffffffef,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0x65a0e008ffffffe9,
                    0xeba8415b23a4d418,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x99ed0724ffffff85,
                    0x88147ee76592dd8d,
                    0xffffffffffffffef,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x65a0e008ffffffe9,
                    0xeba8415b23a4d418,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x99ed0724ffffff85,
                    0x88147ee76592dd8d,
                    0xffffffffffffffef,
                    0x3fffffffffffffff,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xb1adb5ae71ccf9ed,
                    0x09053a2f5ae5290a,
                    0x07eac691aaaca953,
                    0x3b2cea2c7e3ff8e2,
                ])),
                Felt::new(BigInteger256([
                    0x21003e301f188f71,
                    0xd3c21cc70647fd2f,
                    0xb4bdf1701f8c7cce,
                    0x2aaa6d03d3ad5597,
                ])),
                Felt::new(BigInteger256([
                    0xa277769de7c79490,
                    0x565f30f97861a28d,
                    0x1a60fcce7da876ff,
                    0x323f73e08858e30e,
                ])),
                Felt::new(BigInteger256([
                    0x59269c311789c8a7,
                    0x61143fa7891e9b43,
                    0x435c8863ff042513,
                    0x3bfaac1d30824293,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x36d42fc8da89d508,
                    0x8e4fd1ffe60cf81f,
                    0xd3cfc7bb8c6738e3,
                    0x0af14bc125e8c50d,
                ])),
                Felt::new(BigInteger256([
                    0x9f307441eaceff81,
                    0xf282f1db6c6e6afb,
                    0xa8f07bb56969ed5e,
                    0x038bc211dad57e7d,
                ])),
                Felt::new(BigInteger256([
                    0x896f5561d35ab12d,
                    0xa1de589698f4cb3a,
                    0xa2ac7852135ed874,
                    0x1017249430e4e5a2,
                ])),
                Felt::new(BigInteger256([
                    0x7c2769f7905a5a4c,
                    0xcbea980739744f36,
                    0x557f63958b98659b,
                    0x11f0d5085336267f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe630280160e911ba,
                    0xc10b5f2cafa1c490,
                    0x108141313200d7aa,
                    0x1522a529a443ea9f,
                ])),
                Felt::new(BigInteger256([
                    0x9668ef3d1ffab502,
                    0x2eb883597abc7524,
                    0xe0d9fecbe2adce81,
                    0x29e4352d44c330f0,
                ])),
                Felt::new(BigInteger256([
                    0xa5dd676dc1cd474b,
                    0xe89d6fe0bbb9482a,
                    0x963e8a71da1abce9,
                    0x1ef10c4a9a5f5be5,
                ])),
                Felt::new(BigInteger256([
                    0xf857531d71250154,
                    0xa59bb84c80d6da24,
                    0x17ff17249720d484,
                    0x3316ed7824ca92d3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3cc1a75943ba824d,
                    0x37d94d3d08fd3e1c,
                    0x21c0a9731dcad29c,
                    0x21117379d3287546,
                ])),
                Felt::new(BigInteger256([
                    0x651d7d0d31712084,
                    0xbe5dfc3d97e3bb65,
                    0x99ff69e6f7220a47,
                    0x1f5b985ac9835978,
                ])),
                Felt::new(BigInteger256([
                    0xa69db94f68c2a348,
                    0xb7ba3dadbf9c96e6,
                    0x40690580016c6b70,
                    0x32b7c6aebf631a08,
                ])),
                Felt::new(BigInteger256([
                    0xb80e833e8ebcd1c6,
                    0x80ec7ff02dcce33b,
                    0x2be901af4b52308c,
                    0x3e9a8b3c82554673,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x907ab78e18059521,
                    0xa3a8d522c4e469f1,
                    0x713995a8196e35fb,
                    0x3ed290095c95feca,
                ])),
                Felt::new(BigInteger256([
                    0x5b81ccbcf19979ce,
                    0xad7fcd3dce79967c,
                    0x1b1062eab9354cf1,
                    0x37fe2b6c8f04bc45,
                ])),
                Felt::new(BigInteger256([
                    0xe95026e9f2a785ac,
                    0xc6a7938d0be4005a,
                    0x5a0f6d48ba86e7fd,
                    0x3dfbbe68a0afa948,
                ])),
                Felt::new(BigInteger256([
                    0xf3614bde84aa57b6,
                    0x1a5896d04f8ce739,
                    0xfad788ea61a06529,
                    0x09e16b389a4a8a3b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x62521377018d6fc4,
                    0xb8953c7bc8077942,
                    0x206a609131fd2933,
                    0x1202053b6a3f0bfe,
                ])),
                Felt::new(BigInteger256([
                    0x9ec62a58999b5d0a,
                    0xf660a5c8be13718d,
                    0x433bf5d2a38d2e0e,
                    0x28770aa241e6a7b2,
                ])),
                Felt::new(BigInteger256([
                    0xc8a7169b60369387,
                    0x6435dfc2d5892452,
                    0x4dda57936f8910a6,
                    0x0b16f90feaf51c1f,
                ])),
                Felt::new(BigInteger256([
                    0xc448319103c2caac,
                    0xa8f5cff6e43e00cd,
                    0x0f6ea2c01d8c65e9,
                    0x0436a88d6417598d,
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
