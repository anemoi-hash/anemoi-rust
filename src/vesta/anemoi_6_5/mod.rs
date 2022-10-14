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

/// Function state is set to 6 field elements or 192 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 6;
/// 52 elements of the state are reserved for rate.
pub const RATE_WIDTH: usize = 5;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 3;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = 1;

/// The number of rounds is set to 10 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 10;

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
        let beta_y2 = mul_by_generator(&y2);
        *t -= beta_y2;
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
        let beta_y2 = mul_by_generator(&y2);
        *t += beta_y2 + sbox::DELTA;
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    apply_mds_internal(&mut state[..NUM_COLUMNS]);
    state[NUM_COLUMNS..].rotate_left(1);
    apply_mds_internal(&mut state[NUM_COLUMNS..]);
}

#[inline(always)]
fn apply_mds_internal(state: &mut [Felt]) {
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
        // Generated from https://github.com/Nashtare/anemoi-hash/
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
                Felt::new(BigInteger256([
                    0x5565a3cfbdd5de09,
                    0xe6c31164dda2d0b9,
                    0x4a90e89bdb64a64f,
                    0x1e44d809ba2ae3ce,
                ])),
                Felt::new(BigInteger256([
                    0x2b2b8c4216bb1db8,
                    0x59b28759d60c719f,
                    0x39bc01d13d0dc18b,
                    0x076d89946363ceee,
                ])),
                Felt::new(BigInteger256([
                    0x4f173dd6a1d9ff60,
                    0xfe28dddaf42a30d8,
                    0x5e0295cb454fe68c,
                    0x2487e76ab13de4e2,
                ])),
                Felt::new(BigInteger256([
                    0x6abe9aa8f2c7d60f,
                    0xa919f728922c7e3e,
                    0xd0949c369f294970,
                    0x1ef45e7d887cc2d3,
                ])),
                Felt::new(BigInteger256([
                    0x2757a281dc66f60d,
                    0xebb602b1e9f4a9d0,
                    0x72aec7faa15e30b0,
                    0x28f27e049511d3f0,
                ])),
                Felt::new(BigInteger256([
                    0x458ac7d47d3a3205,
                    0x0914974e68d660cb,
                    0x846dee2b7d14f612,
                    0x1942f725bce9ae52,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe178910d8f4512a1,
                    0xa7829c4181658e59,
                    0x5c00da7998f41303,
                    0x3640b04bb37c08fc,
                ])),
                Felt::new(BigInteger256([
                    0xc33068a7364ff530,
                    0x377f4d1bd4138c47,
                    0x3eb787bb1ec5b6c0,
                    0x1ba71d8c9fead422,
                ])),
                Felt::new(BigInteger256([
                    0xf780bd42e4bdaea9,
                    0xab112e66ad06e83a,
                    0xa78f4b6bf3c091ec,
                    0x1e91ec353ca749fc,
                ])),
                Felt::new(BigInteger256([
                    0xb17aa0758accd05a,
                    0x73ac6d6336bc8f4a,
                    0x0d79578569a084fb,
                    0x1e0deee694d3b9ee,
                ])),
                Felt::new(BigInteger256([
                    0xc51a7609f2efb85d,
                    0xcb2f7ceebd6fbc7a,
                    0x007e64c87d8d89ac,
                    0x110e19a974d68dce,
                ])),
                Felt::new(BigInteger256([
                    0xbb9072f5b8707de5,
                    0xf1cba8e2ed04b628,
                    0xd5fe42a1fe821c8f,
                    0x2ef8a54183628bf5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04628c08aa350eae,
                    0x1167f305b65808c2,
                    0x2239cc7f01a28b5d,
                    0x2d4571cd9d15141d,
                ])),
                Felt::new(BigInteger256([
                    0x976c30543f3a3001,
                    0x856ec93c73644ed2,
                    0xa975421c9de36996,
                    0x3416432a97a0fc05,
                ])),
                Felt::new(BigInteger256([
                    0x0773eeb5a7ceab1e,
                    0xed814f4e5e582e70,
                    0x575c43c4afa5276f,
                    0x15fb001128b604aa,
                ])),
                Felt::new(BigInteger256([
                    0x094caa4d6249551d,
                    0x4d90fcd4f8f7d670,
                    0x592ba19dd7a2f37b,
                    0x325db14ce42fe2bc,
                ])),
                Felt::new(BigInteger256([
                    0xbba11b8a9041f240,
                    0x5c26ecaa3564a947,
                    0x297c83c89e1aa26e,
                    0x10e47f138f2ad42e,
                ])),
                Felt::new(BigInteger256([
                    0xec914b88d3c03abf,
                    0x7772fdc334d7d8f2,
                    0x30f1413f547576c2,
                    0x328cce3a6d350858,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd64759eb12ab6a4d,
                    0xae17885a8daeac81,
                    0x9a573a36e1232593,
                    0x06502d6fba2f707a,
                ])),
                Felt::new(BigInteger256([
                    0x6dd77a641966882b,
                    0x019efc3aa5936a84,
                    0x992753d56d98d2d8,
                    0x2116eaaa118c68e3,
                ])),
                Felt::new(BigInteger256([
                    0x4a22aeaab9053422,
                    0xf5a3deb7562827ab,
                    0x130df6f6cadc335f,
                    0x01ec007d1b16c010,
                ])),
                Felt::new(BigInteger256([
                    0x7fc0d947c0c98945,
                    0x8a32dfc0b9140996,
                    0x2704959a35e15f7c,
                    0x09a31281475c8c99,
                ])),
                Felt::new(BigInteger256([
                    0xc96be5eab0612e1b,
                    0x9c42153068c8ff7a,
                    0x9095eb6c592a497f,
                    0x0e65e2f6d3eb46a5,
                ])),
                Felt::new(BigInteger256([
                    0x61bf13555ab2d886,
                    0xe00ad97552aeadd0,
                    0xa3beb55285334a7e,
                    0x195ede70c68ad80b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x97e17a46b37aea30,
                    0x63ab11aa21aeb528,
                    0x87bb7641b3fecdf8,
                    0x1354c8fd492e7f0b,
                ])),
                Felt::new(BigInteger256([
                    0x20513282d9dc627a,
                    0xc0b2d8c695b7cb39,
                    0x5830ffa604050ae2,
                    0x28a19f1e7991837b,
                ])),
                Felt::new(BigInteger256([
                    0x933a5a990f48de5c,
                    0x7fe006bd4e2dba5f,
                    0x2a0879c195e9243b,
                    0x2fd4ba8d1c3782c3,
                ])),
                Felt::new(BigInteger256([
                    0x1bafaf5dd8000642,
                    0x29dedcad3042f34e,
                    0x8e725a98cf6c73ae,
                    0x071af8673a2abab2,
                ])),
                Felt::new(BigInteger256([
                    0xb9efa76478ee4a65,
                    0x1cc5d0d392da2e18,
                    0xc03d7c7c770e1daa,
                    0x1e93f057ceb28299,
                ])),
                Felt::new(BigInteger256([
                    0x98f6a740bfb9a572,
                    0x48eafe38392f04ac,
                    0x118f4a843d79a9bd,
                    0x11c0aa41dcb44999,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x24c3d03b6532c4a6,
                    0xe55bfce6f43fe4ef,
                    0x4412811d23a8dcec,
                    0x030ddfc6793bf0b6,
                ])),
                Felt::new(BigInteger256([
                    0x6ab08b042137eb54,
                    0x31d928b400c52a90,
                    0x6940ead7d5863e7d,
                    0x25c884a882efcb75,
                ])),
                Felt::new(BigInteger256([
                    0xb24998bc3eeefd91,
                    0x4afe91e276899827,
                    0x118dac9834409731,
                    0x18737038d19da68b,
                ])),
                Felt::new(BigInteger256([
                    0x5503b00f0a573ea9,
                    0xeaa91f73ae1e5f2f,
                    0xb615c6f6df93fcd5,
                    0x39f2a552f14a0c44,
                ])),
                Felt::new(BigInteger256([
                    0xf48786bd5b48cfe6,
                    0xa2782feef478173f,
                    0xf816e53a628f1212,
                    0x0bcfe59dddd4f3ad,
                ])),
                Felt::new(BigInteger256([
                    0xaf33c863b018cb0d,
                    0x6288f5ad39bd64e6,
                    0x7e4a3e9128b1e2e5,
                    0x2564ee3d83ed4850,
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
                Felt::new(BigInteger256([
                    0x123bd95299999999,
                    0xb83c0a9bfa40677b,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
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
                Felt::new(BigInteger256([
                    0x311bac8400000004,
                    0x891a63f02652a376,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x36f1bea153183937,
                    0x6a561136b9d07a38,
                    0xa7519c3446736b02,
                    0x3b4c9419cc6327c1,
                ])),
                Felt::new(BigInteger256([
                    0xc4dd8098a86f9f01,
                    0xe6d9326eec181649,
                    0x3207bd6ef002e8da,
                    0x1d80e31c320a78cc,
                ])),
                Felt::new(BigInteger256([
                    0x838f6ee4087eb5f0,
                    0xbcd300d19bea5a9a,
                    0x3b4e2d3224aa470e,
                    0x29affd8252da1ebc,
                ])),
                Felt::new(BigInteger256([
                    0xd4a3497ec88602e8,
                    0x55a07c9b1e8fc04b,
                    0xb961673fc4868721,
                    0x329be06dd5c34b39,
                ])),
                Felt::new(BigInteger256([
                    0x9a96f2f5cde03dba,
                    0x8c0d0842c8372a2d,
                    0xf68022a73cfd7fab,
                    0x1c43730747ef2079,
                ])),
                Felt::new(BigInteger256([
                    0x81b4fe659a0e6316,
                    0xc8f9a3ddaf3559e6,
                    0x15b675bd3121dd0a,
                    0x2f21c64ae94d94a2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x306b7d675fa55fef,
                    0x1289c6dcd6f0d8d6,
                    0x34d6ea0cde4da258,
                    0x2bfa0c684fbe76f8,
                ])),
                Felt::new(BigInteger256([
                    0x83edec7577e8cfea,
                    0x9eac5fade772168c,
                    0x671297bae1cfb0ab,
                    0x09a87e3a5e60fe9f,
                ])),
                Felt::new(BigInteger256([
                    0xa550929e885c68d8,
                    0x3ab91258e27dcb32,
                    0x979510020a4684dc,
                    0x0aa2df9af3ae1608,
                ])),
                Felt::new(BigInteger256([
                    0x5e9b04bb2e59b964,
                    0xf6ac144a6a43cbed,
                    0x64fd0e623458ab70,
                    0x0c3d988e81fd30c9,
                ])),
                Felt::new(BigInteger256([
                    0xc30d81488372a313,
                    0xcb7484f80efaa213,
                    0x6c8f1b7461e2d1d4,
                    0x0504b07f8662f5bf,
                ])),
                Felt::new(BigInteger256([
                    0x31355fb8ff30a799,
                    0xe1e781c85f0c8bff,
                    0xda7eb3ee1ee726bc,
                    0x2e275c4fbd07065a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x61b0280fb70de921,
                    0x12b27ee4f3f0480c,
                    0xa8ed7dacef972229,
                    0x33d99b1668dd47e3,
                ])),
                Felt::new(BigInteger256([
                    0x1d56e5d207d76cd2,
                    0x48873ff1037accd6,
                    0x19d5833c00586b48,
                    0x152116cd4bc6e20f,
                ])),
                Felt::new(BigInteger256([
                    0xa793b22a65e3489e,
                    0x7405b055a145ac0d,
                    0x72ec7ff65dcc6ff5,
                    0x06cc336927b73e06,
                ])),
                Felt::new(BigInteger256([
                    0xc241ff2deab1c800,
                    0x9cd3fb94b57c2612,
                    0xd297475a6a09bfce,
                    0x11431993ff2af5d8,
                ])),
                Felt::new(BigInteger256([
                    0x7fef5f96fcebe74d,
                    0xbb57a94b19975acd,
                    0x69dc022f8d72d4db,
                    0x04d68e0364a902e2,
                ])),
                Felt::new(BigInteger256([
                    0x3fd8a6b2b94e8d21,
                    0xe447922d50d9bf16,
                    0x4141cf96ddb80692,
                    0x207fbb3a4b2dbcbd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf7e1f1081ff5ca88,
                    0x7d3fa869639bc9e8,
                    0x5451ed6bcd2bc0a3,
                    0x21e86d07782b7f7c,
                ])),
                Felt::new(BigInteger256([
                    0x445b8614c007ec95,
                    0x36beb683a0bf6f29,
                    0x29c252cbc411763c,
                    0x1826283813605348,
                ])),
                Felt::new(BigInteger256([
                    0xaf594efb158f0d30,
                    0x2a3584f1cb734f62,
                    0x2d84b0e48c5f9f64,
                    0x2fc15c0a825a713e,
                ])),
                Felt::new(BigInteger256([
                    0x5bd7710e92670dac,
                    0xf4d83ab8021aab76,
                    0xdea780fed6b570f0,
                    0x05b4551763981077,
                ])),
                Felt::new(BigInteger256([
                    0x3671aa0f83ef1934,
                    0x209576704f531fac,
                    0x5f21bd41a0dc42b4,
                    0x0c74e291fa087c3d,
                ])),
                Felt::new(BigInteger256([
                    0x5cb18bab9eef2f0c,
                    0x2e778dd2475ef223,
                    0xc97d311233d81d48,
                    0x0dd8cf17c84a2b9a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xedf61da0c4cd5f0f,
                    0xf3b605b3470be395,
                    0x0fee1824d1842f25,
                    0x35f57cf21ede116c,
                ])),
                Felt::new(BigInteger256([
                    0xdf369deb5a0fbc6a,
                    0xccb8bf903e39f35e,
                    0xf23890c325ec3f9f,
                    0x16e374fc18b39b79,
                ])),
                Felt::new(BigInteger256([
                    0xb0c33b19fa6b76c2,
                    0xb04aa35bbea8eb35,
                    0x6c595ea29c157e74,
                    0x2e2a7f8528afe30d,
                ])),
                Felt::new(BigInteger256([
                    0xb4e31f2f0bf5e5ff,
                    0x49176bb3d217097d,
                    0x457f8dc120a446a8,
                    0x05301ec3791fea46,
                ])),
                Felt::new(BigInteger256([
                    0xc3b7109547294873,
                    0xea7e657096d75656,
                    0xaad9b22aa66e0c99,
                    0x1ea7428dc14f387d,
                ])),
                Felt::new(BigInteger256([
                    0xce1314b8c7a3612d,
                    0x6eed818db0befc83,
                    0x014d421d185a15fc,
                    0x0b51d7988af05df0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbd5e1c72f2abc561,
                    0x1e7b15289e999996,
                    0x240d798c6f6e8738,
                    0x0acbad1acfc90e89,
                ])),
                Felt::new(BigInteger256([
                    0x3c9d8b214a3528b5,
                    0xe8cc248f8b70bf33,
                    0xb128059833b04027,
                    0x34d46f6e653e1190,
                ])),
                Felt::new(BigInteger256([
                    0xb53cef1f733a9500,
                    0xbb6acbc8119f8902,
                    0x5625359ac7fa6aea,
                    0x37487b36cf91a12b,
                ])),
                Felt::new(BigInteger256([
                    0x0ef6cc0d73dc8773,
                    0x6618a29626957745,
                    0x2f987bfd82dfb875,
                    0x152bb160f24de9c0,
                ])),
                Felt::new(BigInteger256([
                    0x5b2ef069dceef911,
                    0x329f7903f4a34901,
                    0x1a2dd27644eed6cd,
                    0x109e1d69360efa8b,
                ])),
                Felt::new(BigInteger256([
                    0x5a7feec32e2a7d0a,
                    0x3923c7b865a77686,
                    0x2734c1f799e1b5cf,
                    0x16646a6ab7ceaaa9,
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
        // Generated from https://github.com/Nashtare/anemoi-hash/
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
                Felt::new(BigInteger256([
                    0xdfaa158798ec6d1b,
                    0x1e120d40703349e6,
                    0x446850a1cce98517,
                    0x1c43c518df2f8708,
                ])),
                Felt::new(BigInteger256([
                    0xa2531a53a9708187,
                    0x8475922a6c6ef45e,
                    0x8a764b4ccdb67b7c,
                    0x1e085795d6752891,
                ])),
                Felt::new(BigInteger256([
                    0xab12113876a6046d,
                    0x1bf9c52c2ca89b16,
                    0xc66e657f3dd2e4a0,
                    0x3fe85ff7aa29a95e,
                ])),
                Felt::new(BigInteger256([
                    0x38d240cefcf0277a,
                    0xad2ec40ddb7bc207,
                    0xb6d8aa44bd89790c,
                    0x3f3c02def77a2a3f,
                ])),
                Felt::new(BigInteger256([
                    0x0a5e175c6f3485db,
                    0xf21384addc401910,
                    0xaa1df05be799d2ff,
                    0x33b36e1d410ab9c3,
                ])),
                Felt::new(BigInteger256([
                    0x906ace83f74dd7a8,
                    0x9a4acb817a6d18de,
                    0xa4ff3cd5a8e33aef,
                    0x353340b3f39c5c55,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x26e4b855716a7d0b,
                    0x43147ec995d717be,
                    0x4569a6caa8515451,
                    0x32383c455dd5c649,
                ])),
                Felt::new(BigInteger256([
                    0x21de5874b386d601,
                    0x68fdf323e78627a2,
                    0x7a862c8c6ab69906,
                    0x3d7a2c5ca870cab2,
                ])),
                Felt::new(BigInteger256([
                    0xe2634485f9e7f756,
                    0x2a65dfd753b9c2ed,
                    0x7b598b7fe2d34295,
                    0x03acee8ce71c84db,
                ])),
                Felt::new(BigInteger256([
                    0xc8ff8db5c8fe8927,
                    0x5eea58c2f51caf72,
                    0x598437232df3418f,
                    0x1f42675bb2a477cd,
                ])),
                Felt::new(BigInteger256([
                    0x692b625146c470b2,
                    0xbbb00c50a33da47b,
                    0x29cb3f0aba8853a1,
                    0x245d54d6045e3c5f,
                ])),
                Felt::new(BigInteger256([
                    0x5b6fab960bb62072,
                    0xb8414db19a78c950,
                    0x1a7fbc2a16a4e580,
                    0x21eaa4fc5af73d7a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x63b3114f9badc12f,
                    0xe46bc698f1c9cd3c,
                    0xada8c8707163f2df,
                    0x07f66df82ce6073d,
                ])),
                Felt::new(BigInteger256([
                    0xdf113aecac40c6e7,
                    0x91b99b9f87606c0f,
                    0x6eb16113afa3e414,
                    0x0444eeac3bb8b1b5,
                ])),
                Felt::new(BigInteger256([
                    0x9062822f8d007581,
                    0x66847dccfa5770ee,
                    0xb69fb9e2f92e6a18,
                    0x0b95becc6e7253a0,
                ])),
                Felt::new(BigInteger256([
                    0x38091f0f225b326c,
                    0x9ce185d69d6f4e18,
                    0x6ae089b18016e115,
                    0x0675ebf27b109d7a,
                ])),
                Felt::new(BigInteger256([
                    0x8ac8056e26fbf92d,
                    0xcda529d1061ed946,
                    0xf9e5bd9d3a953b88,
                    0x1ea5d83271d5e6ef,
                ])),
                Felt::new(BigInteger256([
                    0x9166ad6fd63a0fcd,
                    0x1049ed41ea436b7b,
                    0x4b564c82e765c780,
                    0x225a0a91d66fafda,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd5d62d793007e400,
                    0x93c7116c92a62e84,
                    0xe5157b8ccafcceec,
                    0x38fc777a4f2ac9a4,
                ])),
                Felt::new(BigInteger256([
                    0xaf156dd5fcd622cc,
                    0x08f3c75c20c9a84a,
                    0x989ed164cb6730b6,
                    0x331f5dd45ac5a4b9,
                ])),
                Felt::new(BigInteger256([
                    0xedcc2ef3c95ea531,
                    0x11938ebe55c73c4c,
                    0x26a5d87a303c4191,
                    0x375c35cdff100e2b,
                ])),
                Felt::new(BigInteger256([
                    0x40196b98b0bf8fc2,
                    0x02f333f367fff167,
                    0x9143b06dd70e5136,
                    0x25660a7b18ca9b31,
                ])),
                Felt::new(BigInteger256([
                    0xf762bba1a11e4380,
                    0x69b41a6a53ef58c2,
                    0xcfad27a338e6b071,
                    0x2b01366d0a2a6fe3,
                ])),
                Felt::new(BigInteger256([
                    0x590c2507bc6a1967,
                    0x99409e4a7f23e0ab,
                    0xa28f5de119e2b527,
                    0x395b92b973102763,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x63482632eb320236,
                    0xdedd6a3254761889,
                    0xac719d7ab95b17dc,
                    0x387d19c30bf8b089,
                ])),
                Felt::new(BigInteger256([
                    0x58f36b6c231c243b,
                    0xe53905bf05818c02,
                    0x36a1459e899e8b6f,
                    0x1d686426befe137d,
                ])),
                Felt::new(BigInteger256([
                    0xb1faa137d9bce1f4,
                    0x0c7ca38b976407df,
                    0x677f18a1ce27007d,
                    0x14b8f4dcdd768ad1,
                ])),
                Felt::new(BigInteger256([
                    0x303905ed4a892c04,
                    0x9716d4007ab78642,
                    0x4dd75e4f83ef514e,
                    0x126f6a9a2475ae9a,
                ])),
                Felt::new(BigInteger256([
                    0x27a886db81b4921d,
                    0x56ab1eed85bc7a0d,
                    0xf29577b774e13a2e,
                    0x2db57961c544bd86,
                ])),
                Felt::new(BigInteger256([
                    0x64ef9159773c5def,
                    0x65b6388da98a76b7,
                    0xc44dcda2f6ce5735,
                    0x09dde1cef036eee6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x28110e61b57e72e5,
                    0xbfdb770bc11aaae3,
                    0x97991faa0f188a63,
                    0x231343faf1591fb3,
                ])),
                Felt::new(BigInteger256([
                    0xe7deb01891718c9f,
                    0x242d3c0b4f187bea,
                    0x3bcd9a2f6a574d4c,
                    0x0b6fe4f9c365363b,
                ])),
                Felt::new(BigInteger256([
                    0xc4d5ade625a356ec,
                    0x61b81a6f0b4f8e2e,
                    0x373a920520af8c39,
                    0x0d5d40abed47bbb9,
                ])),
                Felt::new(BigInteger256([
                    0xb9cfb7acf91cb636,
                    0x1dbfd9e3f6837be9,
                    0x3ae0cb04ac88297c,
                    0x2349c6d816bfeb4f,
                ])),
                Felt::new(BigInteger256([
                    0x3befcaf105139074,
                    0x7d471cbdf047fd80,
                    0x3902794d617ca8d8,
                    0x3fb12bd438c613a2,
                ])),
                Felt::new(BigInteger256([
                    0x502ea499dab3448c,
                    0x3e68dc4a8f718b70,
                    0x78ebb9994966b22c,
                    0x2af4edaf34e1d5c2,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 6],
            [
                Felt::new(BigInteger256([
                    0x0ddf286cffffffcd,
                    0x2bef85ca17625bdd,
                    0xfffffffffffffff9,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x34853384ffffffe5,
                    0x628ddd6afd5230a2,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x34853384ffffffe5,
                    0x628ddd6afd5230a2,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x0ddf286cffffffcd,
                    0x2bef85ca17625bdd,
                    0xfffffffffffffff9,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x34853384ffffffe5,
                    0x628ddd6afd5230a2,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x34853384ffffffe5,
                    0x628ddd6afd5230a2,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0x0ddf286cffffffcd,
                    0x2bef85ca17625bdd,
                    0xfffffffffffffff9,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x34853384ffffffe5,
                    0x628ddd6afd5230a2,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x34853384ffffffe5,
                    0x628ddd6afd5230a2,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0ddf286cffffffcd,
                    0x2bef85ca17625bdd,
                    0xfffffffffffffff9,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x34853384ffffffe5,
                    0x628ddd6afd5230a2,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x34853384ffffffe5,
                    0x628ddd6afd5230a2,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xf43dbeab06df2aae,
                    0xac411fd9c35c6285,
                    0xcb7e90130e20f5c6,
                    0x071135f90e8c4afb,
                ])),
                Felt::new(BigInteger256([
                    0x1bf4ee50939b04be,
                    0x83077c5b8c05f963,
                    0xaf06976acfbe77b3,
                    0x39d5fc850874fe73,
                ])),
                Felt::new(BigInteger256([
                    0x06e2d5cf1cb4a778,
                    0xcff5cea4ad5a065f,
                    0xa6ee43f50c18f98f,
                    0x2b439109dc8c7519,
                ])),
                Felt::new(BigInteger256([
                    0x1c80c51d8029e79b,
                    0xdccfed135f70f9e5,
                    0xeac6dc9987b70339,
                    0x26cfe69d46b9b469,
                ])),
                Felt::new(BigInteger256([
                    0x6d3aa725573322df,
                    0x20a08e8c669c06e1,
                    0xe1588089442c6b2e,
                    0x2512bd2c0a09e957,
                ])),
                Felt::new(BigInteger256([
                    0x3fb0ec7c20449c64,
                    0x567a2a0873420be2,
                    0xae6d98e5ec6dd2fb,
                    0x36f06a25304c2767,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0fe4c2d337759041,
                    0x2c349501276f5279,
                    0xff195a4bad92226c,
                    0x00d92d4a461e8d8e,
                ])),
                Felt::new(BigInteger256([
                    0x9c259126067927b8,
                    0x3b829f2a0cd4bc4a,
                    0x28af8cd681283a41,
                    0x0213116289d52945,
                ])),
                Felt::new(BigInteger256([
                    0x959d8a21e4833e8a,
                    0x59afe8fb0220bdd0,
                    0x50effa0197208131,
                    0x3c40484464ba2efc,
                ])),
                Felt::new(BigInteger256([
                    0x3ec7c8fa6a47fb82,
                    0x8a38163ef31acbb4,
                    0x2e5c813d898a64a6,
                    0x37a90e26a5077685,
                ])),
                Felt::new(BigInteger256([
                    0x0cc411113f733ee4,
                    0xe7b14adcea87e071,
                    0x03e00ee4b6ed80ee,
                    0x2293fe9cdc8bd0dc,
                ])),
                Felt::new(BigInteger256([
                    0x8d73637f368add10,
                    0x5ac81913a30bb493,
                    0x44fc2e82e941c938,
                    0x36ffb4862372e323,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0b4bc4c6a0560f06,
                    0x31149d0706933834,
                    0xc8646f082f1211e6,
                    0x398dfb47dfcad2eb,
                ])),
                Felt::new(BigInteger256([
                    0x8869ec0908f0d39a,
                    0x54753e3d534ac517,
                    0xad78caf2feefe96e,
                    0x062816a290da5b16,
                ])),
                Felt::new(BigInteger256([
                    0x61f313aa43a60253,
                    0x6e58fa693aa8df2c,
                    0x899d0528dfc60c8c,
                    0x37aad3518aa9298a,
                ])),
                Felt::new(BigInteger256([
                    0xf131dbdb8e45155f,
                    0x0657a73f9945b43d,
                    0xa7fbf85b476e7336,
                    0x0100a36f63d6ca58,
                ])),
                Felt::new(BigInteger256([
                    0xa8156308a8fe0515,
                    0xcc101b47f9fa225d,
                    0x5b9eba97a26d6874,
                    0x214d7e80af98aa2e,
                ])),
                Felt::new(BigInteger256([
                    0xda832642bb812017,
                    0x4a917939898efd5b,
                    0x97b38a468c66d241,
                    0x020d2f808aadd004,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3dd77df8d53d59e7,
                    0x700af58929f6e7b0,
                    0xdf02c98eaebd93a6,
                    0x15336d863026b399,
                ])),
                Felt::new(BigInteger256([
                    0xe03f034c1bb740bb,
                    0x26f50c9826d80f1e,
                    0x3ef1875487914777,
                    0x00e8e254a540b536,
                ])),
                Felt::new(BigInteger256([
                    0x8066fd61b65c3bf7,
                    0x2fc317511a53d7fe,
                    0x38b0139ef2937ce5,
                    0x0769e905e5aba31d,
                ])),
                Felt::new(BigInteger256([
                    0x43bdb75da79d0ceb,
                    0x12f7ac9c9a1a56bc,
                    0xe8346e4779a0bf13,
                    0x1bc7182a44ce69e3,
                ])),
                Felt::new(BigInteger256([
                    0x5fd24e20d1462bad,
                    0x889a5885b4c04cfc,
                    0x488ef7a98610fba6,
                    0x1f5afd8df92f9f3f,
                ])),
                Felt::new(BigInteger256([
                    0x3cf78e4492c0faa5,
                    0x239df261647dea6b,
                    0x4234d47f0d727894,
                    0x35c7ad55beaef208,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0293aa05c0b57d30,
                    0x7967294e498db06a,
                    0xae458a49b6ab1d8a,
                    0x2cacbbe63799779f,
                ])),
                Felt::new(BigInteger256([
                    0x1d92e1744efe9033,
                    0xbdf86fb33bc27a2f,
                    0xe88e5e4249bca5bd,
                    0x3d82463a1e477a1d,
                ])),
                Felt::new(BigInteger256([
                    0x3df433fd94d31138,
                    0xa0a7bf5a134cc23d,
                    0xfc5871a5f68d033c,
                    0x0c92d9d2d85010fe,
                ])),
                Felt::new(BigInteger256([
                    0x2a8f5b4840aed2af,
                    0x2a9a543972c68362,
                    0x46dad1cccbb19c22,
                    0x0abb39b66a9577ae,
                ])),
                Felt::new(BigInteger256([
                    0x65275f956d9ecc1e,
                    0x6b46498581b33e54,
                    0x3c181ce7ff5c27ec,
                    0x13c070336bc81571,
                ])),
                Felt::new(BigInteger256([
                    0x2a578d0c4a4c6480,
                    0x250a43419a9dbbc5,
                    0xcf108287c323cb6b,
                    0x00d8ab51ef045123,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x442b6d43b43c4781,
                    0x648840fbf3432ee0,
                    0x14c3c44a8907d4fa,
                    0x2e1300e2fb2a5ac8,
                ])),
                Felt::new(BigInteger256([
                    0x5bd538d80320b21f,
                    0xaa5a9e463f2c44da,
                    0xe78b93f31cdd94ce,
                    0x31556c505725008c,
                ])),
                Felt::new(BigInteger256([
                    0xd034e484428d2201,
                    0xde5adec1032f65f0,
                    0x6905ca86d6818d77,
                    0x082d798c676a9076,
                ])),
                Felt::new(BigInteger256([
                    0x242d7024cfd4ec7f,
                    0xac174339a1fc7422,
                    0x303f53859d83a026,
                    0x3cd69db91205cf6b,
                ])),
                Felt::new(BigInteger256([
                    0xfc115967bd56640a,
                    0xc754d68c29f8510a,
                    0xd85229fe098c2a70,
                    0x1b16fbbbdf6781f0,
                ])),
                Felt::new(BigInteger256([
                    0xec03d035ed31cd00,
                    0x00e4affbfde105a9,
                    0xd0d8e320dd5e27e2,
                    0x0cb48fac6780233c,
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
