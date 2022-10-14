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
                    0x633f5eb77dadb5e0,
                    0xe4cf7c09cc2307a5,
                    0xbb96ee5af30cf38f,
                    0x1d9218a50b25fecd,
                ])),
                Felt::new(BigInteger256([
                    0x0ecff16050579a38,
                    0xb855378ba700913d,
                    0xdb49cf170bfd723b,
                    0x357656bbfc824c0c,
                ])),
                Felt::new(BigInteger256([
                    0x4c4be92113845154,
                    0x76aee30e00731aa9,
                    0x2a8bbaf293364f27,
                    0x2207f239332df20f,
                ])),
                Felt::new(BigInteger256([
                    0xd9793e5ceb8403bd,
                    0xe7cb754c8530a963,
                    0x7e5aeee29684bee6,
                    0x3383dce010ba029c,
                ])),
                Felt::new(BigInteger256([
                    0xfe668daa2c260338,
                    0xcfdc703d74d634ce,
                    0x67ed26cef655ee84,
                    0x09281dba30809153,
                ])),
                Felt::new(BigInteger256([
                    0x10749534123b78c3,
                    0xaa6f632d04f909b6,
                    0x966452babc5217ab,
                    0x3d49b5cb2103b7af,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf6abc7c85041a460,
                    0x439647b21c50c33d,
                    0x2ea693d0b322426a,
                    0x02abbdc341dc245a,
                ])),
                Felt::new(BigInteger256([
                    0x12b4fb160f759ebd,
                    0x6d873f5eb9e7ad0a,
                    0x6451f52319642e68,
                    0x0fdd45b4ec319147,
                ])),
                Felt::new(BigInteger256([
                    0x420e839981b39953,
                    0x19f499ce63b83871,
                    0x58c1e04508d300ef,
                    0x2b83305829959a36,
                ])),
                Felt::new(BigInteger256([
                    0xa1039cb1fb841b49,
                    0xfea42d628de7e996,
                    0x2f043a17f25adaa3,
                    0x1434e9844db6e9ef,
                ])),
                Felt::new(BigInteger256([
                    0x75878c79aa8ea73c,
                    0x3d722df7de725827,
                    0x8223feb7ae9e61a9,
                    0x1c57717bac1575a2,
                ])),
                Felt::new(BigInteger256([
                    0xa22fa317c047179c,
                    0xdb95bc9f6207e562,
                    0xe9e8e81d1ec40f0f,
                    0x23e01bfb5f850d7e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3f6a87a96d2f52c4,
                    0x862a99aabec33ee5,
                    0xfa881e5e2598b294,
                    0x311e624a2aa0153b,
                ])),
                Felt::new(BigInteger256([
                    0x024b7f498e696e26,
                    0x98b9d06c4df7115e,
                    0x9fe937fb4db1ff9a,
                    0x21ae4d287c7ac1d8,
                ])),
                Felt::new(BigInteger256([
                    0xef1155e7b4e877a0,
                    0x9e4e1d2cafe5d5c1,
                    0x12ac9b5826d77864,
                    0x306e41750a9e755f,
                ])),
                Felt::new(BigInteger256([
                    0xfa8bcc1953b1d712,
                    0xb053a8eef999fcbf,
                    0x708787252fcc270a,
                    0x04328333e9ae9b22,
                ])),
                Felt::new(BigInteger256([
                    0xb2058e3a669e9272,
                    0x3891ff670519264c,
                    0x9e687c85ff1515ed,
                    0x35e328eda2425857,
                ])),
                Felt::new(BigInteger256([
                    0x61b6fc2679d6654c,
                    0x14b410ab57ad9f97,
                    0x54ff6d2ec2fb662e,
                    0x0a086cad8a581bac,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x48c6e5065052611e,
                    0x0c8ed22f26f4f555,
                    0xd581327968d52e10,
                    0x0131202ab261acf1,
                ])),
                Felt::new(BigInteger256([
                    0x2c74855f5aef0054,
                    0x9fcb30396b33b98a,
                    0x3557c05c4fbb9adf,
                    0x0f78ba27b51d4fc2,
                ])),
                Felt::new(BigInteger256([
                    0x7a021b682a373298,
                    0x039ad2181f61c4ef,
                    0x1a61017be6bc47e0,
                    0x2c7e13c02c08bec4,
                ])),
                Felt::new(BigInteger256([
                    0x1a1826c77cc98b9e,
                    0x2eaeb9bebe18f476,
                    0x5443e7553ad30167,
                    0x39b8db0658d57087,
                ])),
                Felt::new(BigInteger256([
                    0xd4e5f9d89aa72b42,
                    0xf0b3640e6b787dbd,
                    0xfeb50cd234fb81d7,
                    0x1681c1dd630d441d,
                ])),
                Felt::new(BigInteger256([
                    0x81e72a5137b1bf09,
                    0xe208f1bad5796eff,
                    0x3f1357d271af1a2c,
                    0x135dcb9b3ec00b7a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbff5bb2098f47871,
                    0x3353c74128bd4ec9,
                    0x70fe8aff947e45be,
                    0x217c3866b87637e2,
                ])),
                Felt::new(BigInteger256([
                    0x45f1cd35cf326de9,
                    0x6006b6c2ab56bd19,
                    0xe04e43403557663d,
                    0x2b4698eaab3f6942,
                ])),
                Felt::new(BigInteger256([
                    0xa7c709da102f8c8e,
                    0x20411eb97a83eb53,
                    0x42e09f887321fbf4,
                    0x39d8165267b3e00a,
                ])),
                Felt::new(BigInteger256([
                    0x2b31a2f5d663b5a2,
                    0xd0e9fc0558afef42,
                    0x2c88c46f4c755543,
                    0x3bbf45513396a398,
                ])),
                Felt::new(BigInteger256([
                    0xd8d5dd4364e80caf,
                    0x1fa945d5765816b6,
                    0x3079b20db4786f9d,
                    0x064b29e49dfb28ff,
                ])),
                Felt::new(BigInteger256([
                    0x9f68256248ecb551,
                    0x5a0c0a365c30b99a,
                    0xf8a2c5f8d29a0657,
                    0x2298ce1a5221ab3f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc8085f95ef44c253,
                    0xd14d79babf3e1a26,
                    0x66a8e7e0d7c8343b,
                    0x174488c4c79f3ee3,
                ])),
                Felt::new(BigInteger256([
                    0x24281f9e5cc43805,
                    0xe99bd9ee0442a3d4,
                    0xa5c4daf7d61210ff,
                    0x16708dc9aada0dc6,
                ])),
                Felt::new(BigInteger256([
                    0x0ef8460272e03fa8,
                    0xd92760afa5eee8bb,
                    0xcacb1e18a62f63ee,
                    0x3f8a396485a81219,
                ])),
                Felt::new(BigInteger256([
                    0xdf67869b771011f4,
                    0x19bdee9fa80300fe,
                    0xe0182f2f32abb43b,
                    0x097f7895eaca4f6e,
                ])),
                Felt::new(BigInteger256([
                    0xe34c268720d2500b,
                    0xe0353be177340a7b,
                    0x5d007b6375b34322,
                    0x156ad3039fdcd4f8,
                ])),
                Felt::new(BigInteger256([
                    0xc968f477d597dca8,
                    0x609e4908a02e469d,
                    0xc5d7e4c4436c9494,
                    0x24f746ac1ef21dec,
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
                Felt::new(BigInteger256([
                    0x0a7e7c3e99999999,
                    0xb83c0a9bfa6b6a89,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::new(BigInteger256([
                    0x0a7e7c3e99999999,
                    0xb83c0a9bfa6b6a89,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
                Felt::zero(),
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
                    0x3785babe3451890e,
                    0xbad6a04b96e7e335,
                    0xdaecef0c6271ae56,
                    0x309b6fa4569ce2e5,
                ])),
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
                Felt::new(BigInteger256([
                    0x376eae911f9c7044,
                    0xf15d13a83fed92e2,
                    0x8a6f1eb3ded38974,
                    0x0a63a5046381f69e,
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
                    0xe89d156895f33598,
                    0x22cc8f90197cdefa,
                    0x59d1af10ece372fe,
                    0x3c38ccf4eeef1fb8,
                ])),
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
                Felt::new(BigInteger256([
                    0x1fae7c1b952b2763,
                    0x6ff92987f72625f6,
                    0x05ce29c9b82e2059,
                    0x02ee5a311dcf8a77,
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
                    0xae41e60699999981,
                    0x819db2fb1b340ff2,
                    0xccccccccccccccc9,
                    0x0ccccccccccccccc,
                ])),
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
                Felt::new(BigInteger256([
                    0x64b4c3b400000004,
                    0x891a63f02533e46e,
                    0x0000000000000000,
                    0x0000000000000000,
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
                    0x9d1e571fe0ae956e,
                    0x921a1f3a3171023e,
                    0xb2e4deb07d17d6ae,
                    0x13ebe8d4f1f9f4bc,
                ])),
                Felt::new(BigInteger256([
                    0x59997ce713de966b,
                    0x9a57e14ebd893fc8,
                    0xd6a4a097e9c67baf,
                    0x38c57d115383588f,
                ])),
                Felt::new(BigInteger256([
                    0xa0b8220748d6d9b9,
                    0x8e83fcc582c1e04f,
                    0x8758a0151a2f54d9,
                    0x0e1d9ee2eeef9310,
                ])),
                Felt::new(BigInteger256([
                    0x073d2467a5c8e1c3,
                    0xe1cd0c5bd777e43b,
                    0xbf7ea0cd1e7d03fd,
                    0x3975fdee87e176b7,
                ])),
                Felt::new(BigInteger256([
                    0x3d066c21d949d979,
                    0xd1ca5b5d757f5cff,
                    0xa21522be93b1cd6a,
                    0x23f5bf228d4a0a2a,
                ])),
                Felt::new(BigInteger256([
                    0x02a6175252447781,
                    0x050e5054ae69120b,
                    0x1370d53ec207336a,
                    0x2b79b5ab5b4d6410,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe037aaaecf3a2883,
                    0x043861fb3b4fd617,
                    0x9fe1fc9aeeb1632d,
                    0x2c14540269dd6c08,
                ])),
                Felt::new(BigInteger256([
                    0x3093aef990182ef3,
                    0x4a5c9e8505c78654,
                    0xe644898a05852d1c,
                    0x05bd6318aee24f75,
                ])),
                Felt::new(BigInteger256([
                    0x80f1121b4d7c6b3c,
                    0xede835e55d5ea32f,
                    0x203cc0fe61a4885a,
                    0x248e11c282f25d7c,
                ])),
                Felt::new(BigInteger256([
                    0xf31483121a5ec40f,
                    0xb7fd6c70c79161a5,
                    0xdc1a80d4691fde78,
                    0x1f83f0bb26e068d9,
                ])),
                Felt::new(BigInteger256([
                    0xd5a608118aa9f3be,
                    0xfcf18f473581a8ee,
                    0xb391deb40ac5e76e,
                    0x3f641c7bbdf9998b,
                ])),
                Felt::new(BigInteger256([
                    0x50001d7b6f0cb764,
                    0x8f338f5d2918ebef,
                    0x932ebf5f0992c415,
                    0x37f5fed9a6cf4f84,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd2c410313cef3e9c,
                    0x02ee1a8848d974d2,
                    0x7eaa2e7808d053a0,
                    0x20ee3e2c05be8683,
                ])),
                Felt::new(BigInteger256([
                    0x760315440b9d1bce,
                    0x217b562ecfe555cf,
                    0xecf7b4b941af636f,
                    0x1cb66c722c412320,
                ])),
                Felt::new(BigInteger256([
                    0x008b76e782c455e4,
                    0xcc0f94e398253308,
                    0xae2d8fb78d9e3431,
                    0x3794000f844bfed9,
                ])),
                Felt::new(BigInteger256([
                    0x547e76c10ca8f1e3,
                    0x54bfba40b902496e,
                    0x916b3e1be9048d9e,
                    0x019dba2f0024f14e,
                ])),
                Felt::new(BigInteger256([
                    0xf80b00b7f9c4cbbd,
                    0xedafc9f6f7fe84a0,
                    0x003d9c306cc625f8,
                    0x1291a821ed53791b,
                ])),
                Felt::new(BigInteger256([
                    0x91a0a352b9dd9985,
                    0xa8133a44bc649466,
                    0x04787dba3ae4e051,
                    0x295843f08d9037ce,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdeca046f567c8f38,
                    0xfffc807d84581ea4,
                    0x395ab3deff96cfda,
                    0x2f864bc465061cae,
                ])),
                Felt::new(BigInteger256([
                    0x645344f2d452b137,
                    0x18a1ea3c8f4e2667,
                    0xa3084400ca62cb0a,
                    0x34722a71f4e86e0d,
                ])),
                Felt::new(BigInteger256([
                    0x9f009af611464bc2,
                    0x590c1aeb7832722b,
                    0xcfee78cc68144af4,
                    0x11b40504133ecbc5,
                ])),
                Felt::new(BigInteger256([
                    0x8f7f1ffdc9f8c4de,
                    0x497376163fd42b59,
                    0xd836a2639cdc6478,
                    0x37f1d1198b9a72ff,
                ])),
                Felt::new(BigInteger256([
                    0xa8a050fd4a61aa1e,
                    0x75eb16489c0b4d0d,
                    0x13e5af7cf38f9970,
                    0x1f4d2d579ab239fc,
                ])),
                Felt::new(BigInteger256([
                    0x19ea281becb3875e,
                    0x0ff71c4e2f5d3496,
                    0x161a8eea1948f182,
                    0x3024f337b28f08e6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1eee407cf12ab3c0,
                    0x638c4c752d9ca4fd,
                    0x92bea868d549a80c,
                    0x0774e22ab00216e9,
                ])),
                Felt::new(BigInteger256([
                    0x16a55c544f3c172c,
                    0xfff5b3a5c6af6f86,
                    0x4ea11b37c35cca69,
                    0x0ddfdee51e7a45f2,
                ])),
                Felt::new(BigInteger256([
                    0xb627b37ed7be0e49,
                    0x93d7752b9a6af613,
                    0x4f7de4c9b33472ff,
                    0x3344656271c8ddcf,
                ])),
                Felt::new(BigInteger256([
                    0x3400066e0b3f44e2,
                    0x2b09f1c663b7a0b5,
                    0x4a6d338950075d64,
                    0x19e798547089b288,
                ])),
                Felt::new(BigInteger256([
                    0x7b783a399762002d,
                    0xad0902560cdecdd0,
                    0xeb04ff839fa2c542,
                    0x37be575f0935c3b0,
                ])),
                Felt::new(BigInteger256([
                    0x50db7cff74dee7d1,
                    0xfa7b4de2a88cb7d2,
                    0x0f6a193cee2da56c,
                    0x298569b54b74cfe2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0371adbaecd3565c,
                    0x9035f59ba80162ef,
                    0x0e85c2ec7bd7cc41,
                    0x3af783fbc7453755,
                ])),
                Felt::new(BigInteger256([
                    0xe8e7cdbf54483d87,
                    0x3705bdfbde6f7044,
                    0xe3ef666252e86ead,
                    0x0ddf2dc239363822,
                ])),
                Felt::new(BigInteger256([
                    0xadd0b165e78eebfb,
                    0x32dfdbe7fe1f025a,
                    0x4a7c01d783a03636,
                    0x0704c790d7c52b80,
                ])),
                Felt::new(BigInteger256([
                    0x29fbde05807a8bfd,
                    0x675a5d93f6764b78,
                    0x6815b746f768df26,
                    0x35dbb5344a1a7bd4,
                ])),
                Felt::new(BigInteger256([
                    0x086462f05f5becc3,
                    0x5fd3c18825ddbf59,
                    0x5d79322e5a6efb02,
                    0x2c237579e1ff0a03,
                ])),
                Felt::new(BigInteger256([
                    0x3744ef2ab1a99416,
                    0x693d0333d68eed13,
                    0x8c35b141e5a8ac9c,
                    0x11642450a0155603,
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
                    0x97ac24637cdd59a6,
                    0x19c52ceefc422441,
                    0x705d6c14137949bf,
                    0x03cdc8357e9982a4,
                ])),
                Felt::new(BigInteger256([
                    0xaddca7f19f0caadd,
                    0x267001a199801f3c,
                    0x47a85a709175e78b,
                    0x19c42aac377127d2,
                ])),
                Felt::new(BigInteger256([
                    0x70a21f2309d25011,
                    0x5c523e3aee1947e4,
                    0xcb560fe8dbd8efb2,
                    0x2b6d8662d9dfb8ae,
                ])),
                Felt::new(BigInteger256([
                    0x754aaf8b5f2c8573,
                    0x12841454c561f484,
                    0xbcdb029755a521c5,
                    0x21e557fd4ef56dc5,
                ])),
                Felt::new(BigInteger256([
                    0x7d9c1237aca406e3,
                    0x69b9141610117802,
                    0x6317b5c10240aebf,
                    0x27e6b137d9fc65f2,
                ])),
                Felt::new(BigInteger256([
                    0x693be27c005e6be8,
                    0x9324f2032aade96b,
                    0x1d433c9970c3db6f,
                    0x1a88a428902238d4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc7340c6610f25155,
                    0x1dd9b4e25c374083,
                    0x18ca053db306da4c,
                    0x2e6d7102d6350a30,
                ])),
                Felt::new(BigInteger256([
                    0xc587add76fb3c842,
                    0xa9a4986ac03e6fd4,
                    0xffb698e1933e569a,
                    0x112f28200b353fdf,
                ])),
                Felt::new(BigInteger256([
                    0xd25547351ff028f2,
                    0x3f50923601ba6f3e,
                    0x6d78336dcef8e2e8,
                    0x329a4746c438b166,
                ])),
                Felt::new(BigInteger256([
                    0x7f6c3fa6a48997b6,
                    0x6863106e975ea27a,
                    0xc2be3aa3094b6443,
                    0x1c32cd42073eafab,
                ])),
                Felt::new(BigInteger256([
                    0x0e8ef3b2a16f6eae,
                    0x23c15d6227187f71,
                    0xf257d32e32d55280,
                    0x06828b64d52da7f2,
                ])),
                Felt::new(BigInteger256([
                    0x1b7861914ea90f62,
                    0xc1781b768666dbf3,
                    0x0747f162b0f61f2e,
                    0x027f53fe0c80479f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7e17146c66d4afff,
                    0x4d4bd18284096400,
                    0x0b985d0f788e4bfb,
                    0x19528e14f50311ea,
                ])),
                Felt::new(BigInteger256([
                    0x6faa7a8f3325e8ca,
                    0xc4fd756a4966faaf,
                    0xe3d7b96ea39abf86,
                    0x20a09341b89fda85,
                ])),
                Felt::new(BigInteger256([
                    0xe2cf9d51ae65d6ce,
                    0xaf491c92bded47c4,
                    0x8601d4740803787c,
                    0x28b4729765718a28,
                ])),
                Felt::new(BigInteger256([
                    0xdafbbc94e64709ed,
                    0xd6ae7d23045c3e4e,
                    0x23a154e0fb313a79,
                    0x13ace72dea6899f0,
                ])),
                Felt::new(BigInteger256([
                    0xc0121c6437cee8bf,
                    0x2419c48e259e7eff,
                    0x5e3e0c50d6c14edf,
                    0x3c198d0bfc1b1862,
                ])),
                Felt::new(BigInteger256([
                    0xe6508f8e57db323a,
                    0x04bcdffd22723dcf,
                    0x85bdd11dd0aaef30,
                    0x2ace2295c019d1e7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6aa85c118f4fa700,
                    0x27755246ff0173ab,
                    0x76e04b5c86ca819b,
                    0x0ee671657f78e95e,
                ])),
                Felt::new(BigInteger256([
                    0xb1d5b1da8510fd26,
                    0xb923a33f2b604a05,
                    0x3f60b1adfed114a6,
                    0x1979cd94251cf888,
                ])),
                Felt::new(BigInteger256([
                    0xeb3632220b75d024,
                    0xd762cdcf6d4d785f,
                    0x6069f4c55cc5adb3,
                    0x29d2e1f00d53c5fb,
                ])),
                Felt::new(BigInteger256([
                    0x524e85359cb35e1a,
                    0xcde5d90134aa7a63,
                    0xd439e3cbf5b83e8a,
                    0x1832c6ddf536c02d,
                ])),
                Felt::new(BigInteger256([
                    0x46a3c903f46b3e49,
                    0x7bc22afabf19b4de,
                    0x32824d58b6319ed1,
                    0x14ded5a566369953,
                ])),
                Felt::new(BigInteger256([
                    0x137bafb2dfbe694a,
                    0x22579da81707df59,
                    0x7ef6ba14e92bc986,
                    0x1f97c175e71922af,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfd140f04b5e9d2b6,
                    0xd6f315f5b27c97ec,
                    0xd84590659976f590,
                    0x054f901e124cd7de,
                ])),
                Felt::new(BigInteger256([
                    0x94f439e00afda980,
                    0x83173ac5399ad81a,
                    0x37e1b8d1532ec978,
                    0x23428b7908d3457b,
                ])),
                Felt::new(BigInteger256([
                    0xa8adf4991faa6127,
                    0xcb32cb146f1c936f,
                    0x519b27c6e139c5ec,
                    0x3c0ac7a10c00c5da,
                ])),
                Felt::new(BigInteger256([
                    0x9ce7478685280017,
                    0xc9bdb44672f9b3eb,
                    0x4c559230e37135a1,
                    0x3b80ccda5c350876,
                ])),
                Felt::new(BigInteger256([
                    0x8ed52dc0919cbea5,
                    0x70fc15f887f4f4f8,
                    0xe9b2d7f6daceea4a,
                    0x2cc7b4afd00c6900,
                ])),
                Felt::new(BigInteger256([
                    0xfe1dc37fcbd1c250,
                    0x6434771dbf150265,
                    0xf76b67fffc13accc,
                    0x0be84c24fdd6eec3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2852000c1f5349d3,
                    0xd32ae218285cc1a2,
                    0x6c5892674c79bd0a,
                    0x18ec7b936393ac83,
                ])),
                Felt::new(BigInteger256([
                    0xdde577bb9571265d,
                    0x9b18a8f4ea95bea0,
                    0x29d316ea6ec10372,
                    0x22aed57d4eddf073,
                ])),
                Felt::new(BigInteger256([
                    0x103f9508fc4c4ad8,
                    0x0b93dabb72f4f04d,
                    0x83a7f0c59769f4fd,
                    0x18a215f518c65427,
                ])),
                Felt::new(BigInteger256([
                    0xe33ccbcc99256749,
                    0x863240b69032d784,
                    0xd3e2e38708f9e9f0,
                    0x0ffd18e21b72db5b,
                ])),
                Felt::new(BigInteger256([
                    0xeceefb2aef8d1f88,
                    0x3539418c382937e5,
                    0x01e6739ce19f258b,
                    0x0b1b0344b78cdf3e,
                ])),
                Felt::new(BigInteger256([
                    0x01133151ccdf95ee,
                    0xc81904994621b45e,
                    0x1309a8f8a9bf3622,
                    0x0411069ab45d55e5,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 6],
            [
                Felt::new(BigInteger256([
                    0x7bff40c8ffffffcd,
                    0x2bef85ca25aa5f80,
                    0xfffffffffffffff9,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0xd83bd700ffffffe5,
                    0x628ddd6b04e1ba16,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0xd83bd700ffffffe5,
                    0x628ddd6b04e1ba16,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x7bff40c8ffffffcd,
                    0x2bef85ca25aa5f80,
                    0xfffffffffffffff9,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0xd83bd700ffffffe5,
                    0x628ddd6b04e1ba16,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0xd83bd700ffffffe5,
                    0x628ddd6b04e1ba16,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0x7bff40c8ffffffcd,
                    0x2bef85ca25aa5f80,
                    0xfffffffffffffff9,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0xd83bd700ffffffe5,
                    0x628ddd6b04e1ba16,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0xd83bd700ffffffe5,
                    0x628ddd6b04e1ba16,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7bff40c8ffffffcd,
                    0x2bef85ca25aa5f80,
                    0xfffffffffffffff9,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0xd83bd700ffffffe5,
                    0x628ddd6b04e1ba16,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0xd83bd700ffffffe5,
                    0x628ddd6b04e1ba16,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x7afd7964c72aa523,
                    0x61e220acf270c3b2,
                    0xaddd425e2d634033,
                    0x3528023e4a488bc5,
                ])),
                Felt::new(BigInteger256([
                    0xad2bd53d4d0594d5,
                    0xa6fc9ac32059bfa1,
                    0xb0b41610f02bdfc5,
                    0x36b592cff76945e0,
                ])),
                Felt::new(BigInteger256([
                    0x7bae4c191931bb2b,
                    0xe155878b6b97234d,
                    0x44d186bdcead47f8,
                    0x18369a1a8a506db7,
                ])),
                Felt::new(BigInteger256([
                    0xea6816934741b5e5,
                    0x8ca5b59fea47a4d3,
                    0xdcf38eab8026be89,
                    0x1550db6785cd2f24,
                ])),
                Felt::new(BigInteger256([
                    0x65c5cfa588e10e07,
                    0xf29ea0ccf9c23cb1,
                    0x30a1ff4f1f3e3307,
                    0x2bea0d52f4e9c3a3,
                ])),
                Felt::new(BigInteger256([
                    0xedde2969bebf13c6,
                    0x2d2c06d61b33518d,
                    0xc994cbf5d1ac66f1,
                    0x03ef723d2105a455,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfc28eb259502a5e3,
                    0xa426e220a033cc6b,
                    0x2543ece69f3cc5d3,
                    0x175d79d9a9c7a567,
                ])),
                Felt::new(BigInteger256([
                    0x43b15a932056e64d,
                    0x7af6c46affe5f824,
                    0x3bd99f4451219f6f,
                    0x3c9ffd84b685c110,
                ])),
                Felt::new(BigInteger256([
                    0x172c6f56e45f87d9,
                    0xf51b4f1c69d93d38,
                    0xe920e683e1597cfe,
                    0x2beca474fe772436,
                ])),
                Felt::new(BigInteger256([
                    0xa3d402e1f27f35b7,
                    0xa37ee366e14abc25,
                    0x45cc444a19ba67c3,
                    0x12bf67e7370a5557,
                ])),
                Felt::new(BigInteger256([
                    0x74ca31ab26c8749c,
                    0xaa9b99098fbe9591,
                    0xc756e9c0124466ff,
                    0x15ffe1ad05e75dec,
                ])),
                Felt::new(BigInteger256([
                    0xe3af63b51a5fd07e,
                    0xdca1fecfe13ffba2,
                    0x85bd4becb86c1ff2,
                    0x3f3eda383da33f09,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1e037f75b2851192,
                    0xe2d374019d612aaa,
                    0x4d74e283a7055a55,
                    0x2cca974bd75b82f5,
                ])),
                Felt::new(BigInteger256([
                    0xf71adde001f7cacb,
                    0xf59c71da5ddee119,
                    0x8d793cc2443a65f0,
                    0x05795e4ba8da9f3a,
                ])),
                Felt::new(BigInteger256([
                    0xfd65eb37e3b32f90,
                    0x8fedde957f9c4b23,
                    0xa3d35f300665b3eb,
                    0x07f1cc41e720be40,
                ])),
                Felt::new(BigInteger256([
                    0xbf3a1dfd0c5ee23a,
                    0xd339a243d3eae4ca,
                    0x90fa1848bc5a2744,
                    0x0974dbf1272fffd6,
                ])),
                Felt::new(BigInteger256([
                    0x21c5c8140f0d4c97,
                    0xf36b4b4641f70907,
                    0x962285d38f62626f,
                    0x09483387503febfa,
                ])),
                Felt::new(BigInteger256([
                    0x83c4e577552cc7dd,
                    0xe48b36fab4661993,
                    0x80956392fda2b404,
                    0x2afacaff9709e5c3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb72b126f25b1c7f9,
                    0x06d366d986b8f4bf,
                    0x4b1e327954323080,
                    0x2dd1c19571e914a3,
                ])),
                Felt::new(BigInteger256([
                    0xe90575cf4dadb4d7,
                    0xaeb32f9f30fe2c3d,
                    0x9852c4e55577fac4,
                    0x397ea8a9e738bfcf,
                ])),
                Felt::new(BigInteger256([
                    0x7ffb4e7a5d151048,
                    0x1143da79811b1287,
                    0xf22c1f41fd8b4a62,
                    0x0dcce67fafcd4d5b,
                ])),
                Felt::new(BigInteger256([
                    0x4474c15846761398,
                    0x532d519fa86d1674,
                    0xa75fe0f0f0a6f9af,
                    0x2e016c8a0ba93bb5,
                ])),
                Felt::new(BigInteger256([
                    0xc34db0e8e3aa7e13,
                    0x5f09d3b0cadc05f0,
                    0xd69a7a696bf6a10d,
                    0x2d74797117617ce7,
                ])),
                Felt::new(BigInteger256([
                    0x94a2c0224289fecf,
                    0x167b1b96f498efdc,
                    0x4fbc209c6ddc2228,
                    0x2024b48edb60e17d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe0712a050c76e0a8,
                    0x8252eb19cb6405a0,
                    0x332609dc33532e69,
                    0x2b6099f3bea4f7d2,
                ])),
                Felt::new(BigInteger256([
                    0xdf8c1b415f3b61f4,
                    0xa6a74b34e92573ab,
                    0xa82f101952c69ca8,
                    0x14c801bc5723fa9d,
                ])),
                Felt::new(BigInteger256([
                    0x95d948a3b8392834,
                    0x5ec2daaa1bd96a0f,
                    0xc2d8b29433bb5b39,
                    0x39db23b0705442af,
                ])),
                Felt::new(BigInteger256([
                    0xa1f1cad5546e3aaf,
                    0x9013d3bb4cf835c6,
                    0x3b9de4ee71946c54,
                    0x3d9b5562075f978f,
                ])),
                Felt::new(BigInteger256([
                    0x9f95623ff7368163,
                    0x1a84158a57699d6e,
                    0x5eca1aeb4818a33f,
                    0x223401189aec8214,
                ])),
                Felt::new(BigInteger256([
                    0x007a2c1529097b9c,
                    0xd9c4354eb4a39abe,
                    0xd43f3203258f75e1,
                    0x274fa06e6a4a043e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x336d01993b2ea25a,
                    0x283018fe5fff0cb2,
                    0xc9d629f7c6172fa1,
                    0x0c063eb038f9f474,
                ])),
                Felt::new(BigInteger256([
                    0x251aff1aa241e666,
                    0x63999ebe3f213f8d,
                    0x28735d2db04c896e,
                    0x36c5beda2e5141bc,
                ])),
                Felt::new(BigInteger256([
                    0x8564ab272e5de252,
                    0x81f5bc3114c084e1,
                    0xcb35e3b4848ba9a5,
                    0x37ef55535986a32b,
                ])),
                Felt::new(BigInteger256([
                    0xafbf7945010ebed2,
                    0xe810e031e5b01ea7,
                    0x15c1b3d029559308,
                    0x26a1af83a65bb580,
                ])),
                Felt::new(BigInteger256([
                    0xc504f68eba27b9e2,
                    0x7a06f0ba45fc28c0,
                    0x385e8e38b83fed60,
                    0x1f1d8649f5287dee,
                ])),
                Felt::new(BigInteger256([
                    0xebcdb40813c69ade,
                    0x3622f410e5d5aa44,
                    0xf06cce901ad4dbcb,
                    0x0b952fd465908d76,
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
