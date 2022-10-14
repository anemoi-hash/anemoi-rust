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
/// 5 elements of the state are reserved for rate.
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
                    0x27343463dace8d46,
                    0x693fe0a1d19bbf8b,
                    0x4a93d5b80d628027,
                    0x10c69e7f6647d57c,
                ])),
                Felt::new(BigInteger256([
                    0xeccfb90793647203,
                    0xbf2e22a249a6aad7,
                    0x754fef552f5021ea,
                    0x0e0cf730677be856,
                ])),
                Felt::new(BigInteger256([
                    0x21b87f0a6b8b0190,
                    0x6b00ac166986fd78,
                    0x8dfbbbcafeac0000,
                    0x048506946ff9e16b,
                ])),
                Felt::new(BigInteger256([
                    0x1497d179e5d2aaf0,
                    0x97903ef5820904ab,
                    0x4ccbb6aa51392c58,
                    0x0eaa85d051ba2ffc,
                ])),
                Felt::new(BigInteger256([
                    0x107e6d6b7506ee64,
                    0xf4c730ba9a76da6e,
                    0x8e0ffc6dd1f8f729,
                    0x0b357cbe15a46137,
                ])),
                Felt::new(BigInteger256([
                    0x2c5221634b2eac12,
                    0xd2754790d147f995,
                    0xc26027561d77fb37,
                    0x02d17c2c8f7bf902,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x706ab5248b49f842,
                    0x7b2fb716bf733884,
                    0x73bf31acdf710c46,
                    0x0d96077ce156ab7f,
                ])),
                Felt::new(BigInteger256([
                    0xa767a799c7684618,
                    0x0f0c13a221e16ef3,
                    0x190f9988c9002950,
                    0x00531cbd59eac1f0,
                ])),
                Felt::new(BigInteger256([
                    0x312fe6b4f10ffdca,
                    0x9b280bd8379b3f92,
                    0x02a993a06b7527ed,
                    0x0aa83db9bf6b706b,
                ])),
                Felt::new(BigInteger256([
                    0x707e208e7120c4c5,
                    0x0a83988ca901093c,
                    0x286c56b952fb054f,
                    0x0acf517263f33a20,
                ])),
                Felt::new(BigInteger256([
                    0x80f7c39c6588560c,
                    0xdbad13a1dd6fbc0c,
                    0x7b95d49791bb5373,
                    0x06b892cfafb1fa45,
                ])),
                Felt::new(BigInteger256([
                    0x4f41c1b36b4ee458,
                    0xb762b109e142f8f4,
                    0x3b7476209703eb6a,
                    0x08480b65a954e4f9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa36329e0ce208491,
                    0xfc791cfee2da6d8e,
                    0x364cf981610f4fa7,
                    0x0c98968e64063060,
                ])),
                Felt::new(BigInteger256([
                    0x18bf4757569d41e1,
                    0x913e0ab46f82d73e,
                    0xffad8b35e766cdd3,
                    0x0ca8db4fa690dcf7,
                ])),
                Felt::new(BigInteger256([
                    0x2a03815fb39e8f85,
                    0x8e1b578f18406601,
                    0x5b600da380681afa,
                    0x007d08a2749a6dde,
                ])),
                Felt::new(BigInteger256([
                    0x3da7d465fb17791d,
                    0x2c2265aada12e524,
                    0x4b9f1233aee67a5e,
                    0x08ec4b1c8afeac50,
                ])),
                Felt::new(BigInteger256([
                    0x29fbe069e396cfdd,
                    0x2249e27f2007692f,
                    0x60e770f6a8c182d6,
                    0x0c263953d2072404,
                ])),
                Felt::new(BigInteger256([
                    0xa6497e00ba4b8d71,
                    0x91f9f53bd9c2bc8a,
                    0x0c58f2264ea4a500,
                    0x0a007c77a6710b04,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x41978e510d5f0233,
                    0x2d613904bb475e06,
                    0x32c2e778664572e1,
                    0x10a5eb6adbc89cbe,
                ])),
                Felt::new(BigInteger256([
                    0xd87b651a11875615,
                    0x6bdf4942f1e28282,
                    0x1ac70eb44bb1d039,
                    0x0ce6ebd09a752be7,
                ])),
                Felt::new(BigInteger256([
                    0x9529ed70f8965609,
                    0x212be7c48ee1642e,
                    0xf8af23a3981b2fe7,
                    0x059c127ec39665fa,
                ])),
                Felt::new(BigInteger256([
                    0xb60f7eb24ccaae75,
                    0x57396b2068a17790,
                    0xac073d5e9d5102c0,
                    0x0b4e5bd9f21eb2bf,
                ])),
                Felt::new(BigInteger256([
                    0xc57fd49a4e52b5c7,
                    0x84002933164c940b,
                    0xa8fa0e373191b846,
                    0x060d6a0cb4f7f183,
                ])),
                Felt::new(BigInteger256([
                    0x190ca1b908d6b4e5,
                    0x577d55f1f5fdca4a,
                    0x6281826c7d55c1eb,
                    0x02682cd3bf27319b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1d43fd26f48b7e4e,
                    0xb49731ea141e79fa,
                    0x1701f2a21844c40c,
                    0x07c70af2605f544e,
                ])),
                Felt::new(BigInteger256([
                    0x08219123a0e5201e,
                    0x0ef8c2ddb42ef33a,
                    0x911536375f561fe1,
                    0x0b617d5bd800fc9b,
                ])),
                Felt::new(BigInteger256([
                    0x3d0e547de8621748,
                    0x971530b99d848a86,
                    0x4de41a34967a1c81,
                    0x0d7216409d216297,
                ])),
                Felt::new(BigInteger256([
                    0x4df78a28b65bbf34,
                    0x73d1dee0a8a448df,
                    0x887f1af7fbe9fbb1,
                    0x0b8ad462d4b3c456,
                ])),
                Felt::new(BigInteger256([
                    0x2c824662b00d618d,
                    0x7f0c9072f66e6e38,
                    0x6f7321aa5b8c7548,
                    0x05819155ce9bfff6,
                ])),
                Felt::new(BigInteger256([
                    0xaf0c9525965e467f,
                    0x3b84859ffb4a33dc,
                    0x5941cc071a72abec,
                    0x051b86c29627771a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdf872c3d509d0390,
                    0x309612bd00ad0606,
                    0xeb5ca7d0d8c54bc0,
                    0x114d89f73f95ad18,
                ])),
                Felt::new(BigInteger256([
                    0xfe9a4a47be511bf8,
                    0xadb1beed133a9cf6,
                    0x8e7f591892fd23e2,
                    0x0d75bb6ebdd7dd5d,
                ])),
                Felt::new(BigInteger256([
                    0x6ae3ef9612a19c4d,
                    0x8bd68055e7452631,
                    0x343cd9c2a58528bb,
                    0x11fc2e31ac7b7912,
                ])),
                Felt::new(BigInteger256([
                    0xf43f653743966fc1,
                    0x3bd8e5742e8f2017,
                    0x0c79094e2937199a,
                    0x090e6627b35ba9c3,
                ])),
                Felt::new(BigInteger256([
                    0xee74b4077319a5de,
                    0x56d2ffa2d4978c04,
                    0x1110a2fa2f9dbe13,
                    0x04dfc8c3413cc103,
                ])),
                Felt::new(BigInteger256([
                    0xd0b362781d527f26,
                    0xfd8da2eed6b13d31,
                    0x606656b13e447d91,
                    0x0df67e2027742f33,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0xb76f9745d1745d17,
                    0xfed18274afffffff,
                    0xfce619835b36a173,
                    0x068b6ffd78dc8d16,
                ])),
                Felt::new(BigInteger256([
                    0xb76f9745d1745d17,
                    0xfed18274afffffff,
                    0xfce619835b36a173,
                    0x068b6ffd78dc8d16,
                ])),
                Felt::new(BigInteger256([
                    0xb76f9745d1745d17,
                    0xfed18274afffffff,
                    0xfce619835b36a173,
                    0x068b6ffd78dc8d16,
                ])),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x547eaa4fe92c17b5,
                    0xdbdc56d3eeee3424,
                    0xd4e029a17d9b9417,
                    0x0e71b0e9185f0412,
                ])),
                Felt::new(BigInteger256([
                    0x547eaa4fe92c17b5,
                    0xdbdc56d3eeee3424,
                    0xd4e029a17d9b9417,
                    0x0e71b0e9185f0412,
                ])),
                Felt::new(BigInteger256([
                    0x547eaa4fe92c17b5,
                    0xdbdc56d3eeee3424,
                    0xd4e029a17d9b9417,
                    0x0e71b0e9185f0412,
                ])),
                Felt::new(BigInteger256([
                    0xd66a1b1b52ed0c94,
                    0x7350e41fff172b46,
                    0xdee8ec7bf3d076d4,
                    0x08092cd96d87788d,
                ])),
                Felt::new(BigInteger256([
                    0xd66a1b1b52ed0c94,
                    0x7350e41fff172b46,
                    0xdee8ec7bf3d076d4,
                    0x08092cd96d87788d,
                ])),
                Felt::new(BigInteger256([
                    0xd66a1b1b52ed0c94,
                    0x7350e41fff172b46,
                    0xdee8ec7bf3d076d4,
                    0x08092cd96d87788d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbc9fbddce41f262d,
                    0xdd9574526d0633f2,
                    0x74ab4b549cae7cc8,
                    0x00f4b6647a675f0c,
                ])),
                Felt::new(BigInteger256([
                    0xbc9fbddce41f262d,
                    0xdd9574526d0633f2,
                    0x74ab4b549cae7cc8,
                    0x00f4b6647a675f0c,
                ])),
                Felt::new(BigInteger256([
                    0xbc9fbddce41f262d,
                    0xdd9574526d0633f2,
                    0x74ab4b549cae7cc8,
                    0x00f4b6647a675f0c,
                ])),
                Felt::new(BigInteger256([
                    0x86ef9156cec705ed,
                    0x31911e672304df9a,
                    0xdb1cd013ebe897a2,
                    0x110866f6f86e425d,
                ])),
                Felt::new(BigInteger256([
                    0x86ef9156cec705ed,
                    0x31911e672304df9a,
                    0xdb1cd013ebe897a2,
                    0x110866f6f86e425d,
                ])),
                Felt::new(BigInteger256([
                    0x86ef9156cec705ed,
                    0x31911e672304df9a,
                    0xdb1cd013ebe897a2,
                    0x110866f6f86e425d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x53e71745d1745bdc,
                    0xaa1116eabffffeb8,
                    0xff0b3527e2b10fca,
                    0x0da5b495c3ed1bcd,
                ])),
                Felt::new(BigInteger256([
                    0x53e71745d1745bdc,
                    0xaa1116eabffffeb8,
                    0xff0b3527e2b10fca,
                    0x0da5b495c3ed1bcd,
                ])),
                Felt::new(BigInteger256([
                    0x53e71745d1745bdc,
                    0xaa1116eabffffeb8,
                    0xff0b3527e2b10fca,
                    0x0da5b495c3ed1bcd,
                ])),
                Felt::new(BigInteger256([
                    0x8cf500000000000e,
                    0xe75281ef6000000e,
                    0x49dc37a90b0ba012,
                    0x055f8b2c6e710ab9,
                ])),
                Felt::new(BigInteger256([
                    0x8cf500000000000e,
                    0xe75281ef6000000e,
                    0x49dc37a90b0ba012,
                    0x055f8b2c6e710ab9,
                ])),
                Felt::new(BigInteger256([
                    0x8cf500000000000e,
                    0xe75281ef6000000e,
                    0x49dc37a90b0ba012,
                    0x055f8b2c6e710ab9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x99d1e2f6e9f7da19,
                    0x6e92820da473e9c1,
                    0xe24a0528998793ba,
                    0x07bfbff6c3b9fa17,
                ])),
                Felt::new(BigInteger256([
                    0x88d400b419a3eca0,
                    0x11a7f5e8f5acb692,
                    0xb8b242369c4d96a7,
                    0x084f9ef227b811a0,
                ])),
                Felt::new(BigInteger256([
                    0xfb83f06e5900f6f0,
                    0x4434a93abeaca609,
                    0x0f577598cb6b7d92,
                    0x0a6e012c89e1b876,
                ])),
                Felt::new(BigInteger256([
                    0xc772216a14bbe2a9,
                    0xa2357e0f90677b91,
                    0x417cc00c37685b3e,
                    0x0431d455917ab824,
                ])),
                Felt::new(BigInteger256([
                    0xf6a6712b42d05dd9,
                    0xdea7761c57a4875d,
                    0x434e146f7dc63c5a,
                    0x0dad5893f936df76,
                ])),
                Felt::new(BigInteger256([
                    0x68e96693daec8bec,
                    0x94959ec86d8a4d3d,
                    0x2b5e642d83c5c073,
                    0x0fd41b19aa92866a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x15b52e6efe7aff08,
                    0x6e4461cfe8d03d73,
                    0xc4c1198503ceafd1,
                    0x081e7c49d20a8874,
                ])),
                Felt::new(BigInteger256([
                    0x689b2993ab5231d8,
                    0x3e741ec78561b61f,
                    0x8d2718811865a50e,
                    0x0f5653ca2dffd01e,
                ])),
                Felt::new(BigInteger256([
                    0x4f7184eac819b4ca,
                    0x255736f1210c0a82,
                    0x5971b48c05fbed74,
                    0x03efb953ec3c497f,
                ])),
                Felt::new(BigInteger256([
                    0xac0f107db86a6b50,
                    0x39882a93f5484860,
                    0x66ea0fb1e50df0a1,
                    0x11b8f856f9201a57,
                ])),
                Felt::new(BigInteger256([
                    0xd9af9287757a70a8,
                    0x07e4627479707ba3,
                    0x372fbd7691801c30,
                    0x000ad54faaa2cd1a,
                ])),
                Felt::new(BigInteger256([
                    0x01434f1a85087aaa,
                    0x2f746c4336591362,
                    0x5493650460a36303,
                    0x0a7767224a15bf02,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8e763d41c394eba4,
                    0xae49342c41482b96,
                    0xaa9fc9c9b890db2a,
                    0x03ccc6ae434e24c2,
                ])),
                Felt::new(BigInteger256([
                    0xd2889dd01518b4d8,
                    0x6a7d6e6d5d89c33c,
                    0x1ee6c6c878a22005,
                    0x0ceaadd96547db8a,
                ])),
                Felt::new(BigInteger256([
                    0x1ea6cd8d5c761696,
                    0x0057500bcdcfb153,
                    0xa17eec749aa7ee30,
                    0x08fcf3014130ed62,
                ])),
                Felt::new(BigInteger256([
                    0xd23305a519ff9ede,
                    0xa98056b97e3c2364,
                    0x6078252451be7d5a,
                    0x0a224f57049c6958,
                ])),
                Felt::new(BigInteger256([
                    0xfa5b6e33c9288095,
                    0x92c2acb0b478755c,
                    0xd9b43c18a66bd76d,
                    0x0756e0e242d28881,
                ])),
                Felt::new(BigInteger256([
                    0x0e48048bd408e799,
                    0x05aa0d1028d59832,
                    0x9d27bee4be0e506d,
                    0x020999d7af561e1c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04f4d61116c919b5,
                    0x0c701279bc112c0e,
                    0xe759da817f3c78c7,
                    0x0dd8f6a0c77cf180,
                ])),
                Felt::new(BigInteger256([
                    0x5e99a435f3264b30,
                    0xcfc47982ce32a88b,
                    0x6c00d71e41b36dfe,
                    0x117dc6d33a134883,
                ])),
                Felt::new(BigInteger256([
                    0x8d337f7cc92cae63,
                    0x04c16c544184e65d,
                    0xff2b62bd5d607d8c,
                    0x04d0e3bda6b8e995,
                ])),
                Felt::new(BigInteger256([
                    0x95acde2bce6248d4,
                    0x4c838a3901560dbd,
                    0x96675f88447420fc,
                    0x0fd8cfdd1b001dd7,
                ])),
                Felt::new(BigInteger256([
                    0x8abc1dc0e189d4de,
                    0x54b44db3e4603bc6,
                    0x57bbb5cf5df578bb,
                    0x047fda13a44151a0,
                ])),
                Felt::new(BigInteger256([
                    0x74e21615991cba27,
                    0x18a218ef42a03c69,
                    0x991d00429cad87ef,
                    0x0c87a7c4ef90a70a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xaaf6dc1d64e3fdcc,
                    0xfdee549316b20bcd,
                    0xf5e9683646a396a5,
                    0x090ef5368b450495,
                ])),
                Felt::new(BigInteger256([
                    0x73a831438b37f2c3,
                    0xefea0edad5722182,
                    0x52e34136159942ac,
                    0x0dc4a0c8ea60467a,
                ])),
                Felt::new(BigInteger256([
                    0x21c64889e5e1d098,
                    0xc31cdba1aa7a6072,
                    0x5d1d07eeaacf5a68,
                    0x016cce5d30672625,
                ])),
                Felt::new(BigInteger256([
                    0xc1342d06f18d5a27,
                    0x97942d2238cb33ce,
                    0x7db1cf193fc189b5,
                    0x0773226937b66205,
                ])),
                Felt::new(BigInteger256([
                    0x4bd5385ca8b586d2,
                    0x9c52b860f714dcf6,
                    0xc546b9a1471243b0,
                    0x0b2e902c683df42c,
                ])),
                Felt::new(BigInteger256([
                    0x7966439c2c6e10f1,
                    0x7d6744ab694f32b7,
                    0x6a6c9f6942010b10,
                    0x0bb710c6bdcead62,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x84e21ced9b988fc0,
                    0x91b2f3be3320bdb5,
                    0xb8545b36ae6940af,
                    0x0aded52630a64cc4,
                ])),
                Felt::new(BigInteger256([
                    0xbb0999efeaddc075,
                    0xe826e460af72bc40,
                    0x53caa91198bf45b8,
                    0x0af5ea0728141701,
                ])),
                Felt::new(BigInteger256([
                    0x4ba1f69926566ff3,
                    0x2dfdfe71bd4e0ffa,
                    0xb0bf8a3696e0c281,
                    0x0ad2b7d3ff5f3230,
                ])),
                Felt::new(BigInteger256([
                    0xa9371a8fa31b9e8e,
                    0x781be975ae6f0387,
                    0x80ff17a20cba65d0,
                    0x0ed01f5766d6a82d,
                ])),
                Felt::new(BigInteger256([
                    0x6a9b66895e8b9adb,
                    0x4cb7bba7fbdaff06,
                    0x42b1e3cba5cb052d,
                    0x01e73d81370f4e88,
                ])),
                Felt::new(BigInteger256([
                    0x783f503edfe131c5,
                    0xe42d87dc3c94b935,
                    0x33e7f74c2f4d2658,
                    0x0520878b3b18d38e,
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
                    0xa92411e63e4e1bee,
                    0xcf89c7814b3d881c,
                    0xe22c6b6437ce49c3,
                    0x06c44cff77b617b9,
                ])),
                Felt::new(BigInteger256([
                    0x103f6ae4261ba307,
                    0x0ba9fc09b1af3319,
                    0x21a61da7cfbd3521,
                    0x0d9f591220550288,
                ])),
                Felt::new(BigInteger256([
                    0x407695dbab6a7b47,
                    0x9e62504dd04cc40f,
                    0x981d226a84342924,
                    0x00b2d5059714839a,
                ])),
                Felt::new(BigInteger256([
                    0xea5274029cf37e80,
                    0x65cbea409ab0eaf9,
                    0x87a87f41eeed17ec,
                    0x06b92ccbfbd1947a,
                ])),
                Felt::new(BigInteger256([
                    0x6d9391d5ee29b8f5,
                    0xfd7c8a2b889c1be4,
                    0x412224a100d7d930,
                    0x02ef171d55a7b903,
                ])),
                Felt::new(BigInteger256([
                    0x7f263a8bac1c25c3,
                    0x91988813db63e521,
                    0xa6311ad50330cafa,
                    0x02f107165c8e927a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa51d3febdd3ed931,
                    0x066ab413d3b1953b,
                    0x6586bd31b94d7890,
                    0x0190ef088d857ef0,
                ])),
                Felt::new(BigInteger256([
                    0x3ffe207942f1673c,
                    0x14e974e399cbc818,
                    0xe5da11a12658e757,
                    0x0b4096015c9ecfe6,
                ])),
                Felt::new(BigInteger256([
                    0xf991a733d20050a7,
                    0xd650cc2d60ecf0f7,
                    0x98a9ee3c25e0b6dc,
                    0x021e1fa614445ef1,
                ])),
                Felt::new(BigInteger256([
                    0xab95afd751693921,
                    0x0470008ce64e908d,
                    0x2a06288156081f32,
                    0x1257d6c40552c2cb,
                ])),
                Felt::new(BigInteger256([
                    0xfad98c4f202d7f1e,
                    0x513df81a7fb831e4,
                    0x3066018eedcff194,
                    0x126bfd0711f62fee,
                ])),
                Felt::new(BigInteger256([
                    0xa78261c2e5aa1389,
                    0x11a7fa00a523e372,
                    0xa46ce93b3cee77e9,
                    0x089a2b29264786f9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2dca13a3abb5c3e6,
                    0xaa3ed1be2e6f44f4,
                    0x91844f43a1f9256d,
                    0x0d78c98a903985fd,
                ])),
                Felt::new(BigInteger256([
                    0xefca8b4c84541d00,
                    0x4905ca3fe2217687,
                    0x29267ee4ed15bae1,
                    0x0dcaab0087238d6c,
                ])),
                Felt::new(BigInteger256([
                    0xb9da3d9cd9640aec,
                    0x1bc213fb2cd48427,
                    0x0448f21643bc4b1c,
                    0x115c6f6ca5880b10,
                ])),
                Felt::new(BigInteger256([
                    0x64987feb82471285,
                    0xb7c983d0ecc4e7c9,
                    0xfdf64437a3268e73,
                    0x06720a0f3b63c9ae,
                ])),
                Felt::new(BigInteger256([
                    0xcb541f0a5d6c1c3a,
                    0xffb5af3a56c7bf40,
                    0x0e94151015270b38,
                    0x0640f97b2f403c80,
                ])),
                Felt::new(BigInteger256([
                    0x388750b3a0cc2466,
                    0x84d8f78bbb7e5541,
                    0x73b00ccdc40bca8c,
                    0x09d73af18608121c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfb0e06816f8de290,
                    0x34479762c6154bd8,
                    0xf652bfead9ea070d,
                    0x026f440cb3fc0230,
                ])),
                Felt::new(BigInteger256([
                    0x833bed19feeb24ac,
                    0x1c2d40f440d66cbd,
                    0xb93491e12a124c17,
                    0x0e2a65f3fad1d053,
                ])),
                Felt::new(BigInteger256([
                    0x634f64397b6184e6,
                    0x669a2e4d8b2a20a2,
                    0x38bcb461ae6d4a59,
                    0x06160f923114d098,
                ])),
                Felt::new(BigInteger256([
                    0x8b30fca9d874acce,
                    0x32559356e01e1150,
                    0xc952efd8ea09375b,
                    0x01983092f85fcbc0,
                ])),
                Felt::new(BigInteger256([
                    0x9e50a7a52dfe9c80,
                    0x78c00a924b5fa529,
                    0x73669b57d1e9c671,
                    0x0afd33e0e4678e2c,
                ])),
                Felt::new(BigInteger256([
                    0x8325e806646053d0,
                    0xdefd119ec61139ff,
                    0xdbda6adf07956347,
                    0x046725d34df60f9c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x588ee4e328c06ba2,
                    0xd8924b1216e53762,
                    0x2f8ed27afdac2869,
                    0x0a8a5b3e4dfd07ce,
                ])),
                Felt::new(BigInteger256([
                    0x969331f7fe3844bd,
                    0x860feeab54e7d4b9,
                    0x4ebd0976e568b071,
                    0x0f237f9df720f70a,
                ])),
                Felt::new(BigInteger256([
                    0x531677b72ef31007,
                    0x56e67b4f331c9483,
                    0xb2c4f7d0206a1fbf,
                    0x0547e996a1576ee9,
                ])),
                Felt::new(BigInteger256([
                    0xa68482e1d45f0a55,
                    0x84ab4045b30fc144,
                    0x26e73624227109ba,
                    0x0999592bf1633a5c,
                ])),
                Felt::new(BigInteger256([
                    0x8f8d90fc8ce2d29c,
                    0xed7d702c81bca89b,
                    0x75bf93a6e61ff713,
                    0x083ae309b01ff254,
                ])),
                Felt::new(BigInteger256([
                    0x67c3ff19d269e992,
                    0xaf097ebca75d629c,
                    0x0938df7fb2768ab7,
                    0x01537068f56d8580,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x92968d2b2ba2b0f1,
                    0xb9b03be243f9acb2,
                    0xabb1ba50d3ededea,
                    0x0b4b9b54efd6669e,
                ])),
                Felt::new(BigInteger256([
                    0x36a1f390f5a4e843,
                    0x6637ce0bfc61f60c,
                    0xaa5d59afc9c4e31f,
                    0x031fe0fd62fc2040,
                ])),
                Felt::new(BigInteger256([
                    0xd46985f83adf9993,
                    0x1e050a2905e55a8a,
                    0xd41ab6ff90b360af,
                    0x0c2d243096513705,
                ])),
                Felt::new(BigInteger256([
                    0x89b60ec281c0906f,
                    0xda47db5b855ffce9,
                    0x1d02a43d08eefafd,
                    0x087e19221f6765e1,
                ])),
                Felt::new(BigInteger256([
                    0xf2a20034a36d0a7b,
                    0x68f9effe0cb62aee,
                    0xb704eda6d079d430,
                    0x0928d9adb2880f2b,
                ])),
                Felt::new(BigInteger256([
                    0x7064fe474ecde120,
                    0x9947e81a1a1b67c4,
                    0x90651fccdd71ec3b,
                    0x00a85511c9bcdfd5,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 6],
            [
                Felt::new(BigInteger256([
                    0xabf9fffffffffd7c,
                    0x6f2ca6fcbffffd62,
                    0xba6dffa003e93c9a,
                    0x08d4fe0427b012b4,
                ])),
                Felt::new(BigInteger256([
                    0x0f827ffffffffeb7,
                    0xc3ed1286affffeaa,
                    0xb848e3fb7c6ece43,
                    0x01bab96bdc9f83fd,
                ])),
                Felt::new(BigInteger256([
                    0x0f827ffffffffeb7,
                    0xc3ed1286affffeaa,
                    0xb848e3fb7c6ece43,
                    0x01bab96bdc9f83fd,
                ])),
                Felt::new(BigInteger256([
                    0xabf9fffffffffd7c,
                    0x6f2ca6fcbffffd62,
                    0xba6dffa003e93c9a,
                    0x08d4fe0427b012b4,
                ])),
                Felt::new(BigInteger256([
                    0x0f827ffffffffeb7,
                    0xc3ed1286affffeaa,
                    0xb848e3fb7c6ece43,
                    0x01bab96bdc9f83fd,
                ])),
                Felt::new(BigInteger256([
                    0x0f827ffffffffeb7,
                    0xc3ed1286affffeaa,
                    0xb848e3fb7c6ece43,
                    0x01bab96bdc9f83fd,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0xabf9fffffffffd7c,
                    0x6f2ca6fcbffffd62,
                    0xba6dffa003e93c9a,
                    0x08d4fe0427b012b4,
                ])),
                Felt::new(BigInteger256([
                    0x0f827ffffffffeb7,
                    0xc3ed1286affffeaa,
                    0xb848e3fb7c6ece43,
                    0x01bab96bdc9f83fd,
                ])),
                Felt::new(BigInteger256([
                    0x0f827ffffffffeb7,
                    0xc3ed1286affffeaa,
                    0xb848e3fb7c6ece43,
                    0x01bab96bdc9f83fd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xabf9fffffffffd7c,
                    0x6f2ca6fcbffffd62,
                    0xba6dffa003e93c9a,
                    0x08d4fe0427b012b4,
                ])),
                Felt::new(BigInteger256([
                    0x0f827ffffffffeb7,
                    0xc3ed1286affffeaa,
                    0xb848e3fb7c6ece43,
                    0x01bab96bdc9f83fd,
                ])),
                Felt::new(BigInteger256([
                    0x0f827ffffffffeb7,
                    0xc3ed1286affffeaa,
                    0xb848e3fb7c6ece43,
                    0x01bab96bdc9f83fd,
                ])),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xb286fd5025b138c1,
                    0xc4e1f0afdb1c0b01,
                    0xb7eb2529760057f6,
                    0x114cd73308f72715,
                ])),
                Felt::new(BigInteger256([
                    0x3982ddab1f90570e,
                    0x1dfc333c13859484,
                    0xb59f311507cf5809,
                    0x11168f2df9a1c434,
                ])),
                Felt::new(BigInteger256([
                    0x89438a892c3c84ba,
                    0xb28fb97d7945a996,
                    0x23f211bc3dee350e,
                    0x0dd5a1172faa6569,
                ])),
                Felt::new(BigInteger256([
                    0xebfec1002bbc213a,
                    0x491d69d9474f830e,
                    0x27f236cc3b46348e,
                    0x0106609d745cc8bb,
                ])),
                Felt::new(BigInteger256([
                    0xbf43c49b1732bdb0,
                    0x81477bd62f343277,
                    0x8a2bc62daaab3271,
                    0x046eccc684d3e352,
                ])),
                Felt::new(BigInteger256([
                    0xb5f2b6f0c0a5894e,
                    0xb318ed15c37f35b9,
                    0xa4abd891f0037d18,
                    0x1234004be6441d39,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4f5d6452019e299f,
                    0x296da7c54409d6b6,
                    0xd8b3f6e86668ba18,
                    0x02a1ecd8e2e28484,
                ])),
                Felt::new(BigInteger256([
                    0x396b3ed92c372ec4,
                    0x79464fe151da12a0,
                    0x47de60a30c4f06dd,
                    0x03660d33d97e8797,
                ])),
                Felt::new(BigInteger256([
                    0x55f045f218586217,
                    0xc510ccc78bfb8c2e,
                    0x76afa7e680729a91,
                    0x0a7c75a66602ccd4,
                ])),
                Felt::new(BigInteger256([
                    0xd66d49371a34a105,
                    0x015266403fbf5b90,
                    0xc473225b1f8e097e,
                    0x0e1160c723fc4571,
                ])),
                Felt::new(BigInteger256([
                    0x79a68a9304e27b66,
                    0xb6354d523f9c816c,
                    0xc127782d467047aa,
                    0x012c8188d357999a,
                ])),
                Felt::new(BigInteger256([
                    0xfa37a066fafc3927,
                    0x031a9befd546bd95,
                    0x4704464ab9d489bb,
                    0x02d3a508dcbf8d7a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x40f4d81879a5b1b8,
                    0xc3ccc414353887e2,
                    0x43d3eb76e01eb70e,
                    0x0d81bd2b78e70ad2,
                ])),
                Felt::new(BigInteger256([
                    0x42e66a6adea2d119,
                    0xfaf690acdad416d0,
                    0x2a254894d0abe49c,
                    0x1125b71eab6476b4,
                ])),
                Felt::new(BigInteger256([
                    0xedd8f8fa1f56fd9f,
                    0x11dbfda73c85e799,
                    0x42d521c6fc888d53,
                    0x0a26b10d54a6218a,
                ])),
                Felt::new(BigInteger256([
                    0xe5b198cab9e5577f,
                    0x669f1da1cd21571a,
                    0x88d940598f8998fb,
                    0x0336367777fa4af0,
                ])),
                Felt::new(BigInteger256([
                    0x586a6dfb3053d806,
                    0x828c42c3eb31ffc8,
                    0x4fcb95b2fcc593ac,
                    0x0889e6c6fe7678f3,
                ])),
                Felt::new(BigInteger256([
                    0xc5ce7b832a5da3df,
                    0x68ebd2699d6dac93,
                    0xacbdb77456cfcfdb,
                    0x04838aa1ff8be419,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8ee003e51a6d703b,
                    0x2c22e6d89d892bc7,
                    0xcbd8b47373860c40,
                    0x04c78e29ef69a5b0,
                ])),
                Felt::new(BigInteger256([
                    0xb690908c08da72f8,
                    0x54611b0a7c8a8682,
                    0x8a1c693e1fa336c8,
                    0x0123d59c153294e6,
                ])),
                Felt::new(BigInteger256([
                    0x5b8b6073107e21ef,
                    0xf3ef0cc261d51202,
                    0xfaf0dd167df5218e,
                    0x11ce1e81d508e11d,
                ])),
                Felt::new(BigInteger256([
                    0xa6c52b1ff8bbe9c3,
                    0xfcf148a37e5c9ee9,
                    0xa5f567788326e096,
                    0x0dc83bb11b4173dd,
                ])),
                Felt::new(BigInteger256([
                    0x038946442c65ca02,
                    0xf7c2d7aab4065c13,
                    0xdaf9089e3dda8b8c,
                    0x0d21bb98563fd5ad,
                ])),
                Felt::new(BigInteger256([
                    0x2661cce230b67391,
                    0xe42b829790677cd6,
                    0xa6d6c9b94ce1b851,
                    0x050da5ec12f7adcc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x391083d5df5860d8,
                    0xc8987580bb112647,
                    0x162e2fd7650f6c09,
                    0x05a3c35b6c3ab6d0,
                ])),
                Felt::new(BigInteger256([
                    0xcc95e0992fdc10f2,
                    0x62c791942041cf5d,
                    0x364b0c0026adc343,
                    0x0b2c2837eb690191,
                ])),
                Felt::new(BigInteger256([
                    0x030dd534adb494a3,
                    0xebe0d297efb72ba3,
                    0x2ea02d4e23ca5937,
                    0x099b1ac177f2ac49,
                ])),
                Felt::new(BigInteger256([
                    0x65e2c8148f54c323,
                    0x3c0b1f1985bae5aa,
                    0xccb660218ab47d26,
                    0x00aa2319386053f1,
                ])),
                Felt::new(BigInteger256([
                    0xd7f44f7e9f779f71,
                    0x28eb57f39c74a712,
                    0xaf17c9f397e8c7c4,
                    0x0f5ea228c4296109,
                ])),
                Felt::new(BigInteger256([
                    0xffc3f7afc2470d45,
                    0x1bd3bce160a3a134,
                    0x878bc2cc0179f01b,
                    0x054a56b803d11dbe,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x53a82bbf2b599a02,
                    0x9f319e320f6b9c72,
                    0x314ecb7861f402ca,
                    0x018e9a44f97c8f3f,
                ])),
                Felt::new(BigInteger256([
                    0x73438411307ecbc7,
                    0x7359f08692116a9d,
                    0xe5ca492fa5d9d002,
                    0x0241a7f4352f914f,
                ])),
                Felt::new(BigInteger256([
                    0x17069b3ef07fb67e,
                    0x920f7bb779bc27e1,
                    0xf5e1dbf885df15e7,
                    0x026cd34e27491e2f,
                ])),
                Felt::new(BigInteger256([
                    0xc8ded67ba5e6cc11,
                    0x49356a3f2a18fc21,
                    0xae48e7c7d54b1947,
                    0x0e98d4fb0096d3e6,
                ])),
                Felt::new(BigInteger256([
                    0xd3fd433318c7551b,
                    0x43c60c017d114eba,
                    0xfe992782d848722f,
                    0x09f561fc29693af9,
                ])),
                Felt::new(BigInteger256([
                    0x6546918fdbed5816,
                    0x9eb74657c723152b,
                    0x4014e011da749353,
                    0x054bc810a0ea78c2,
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
