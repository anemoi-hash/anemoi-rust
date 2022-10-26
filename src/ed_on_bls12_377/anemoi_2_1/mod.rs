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

/// Function state is set to 2 field elements or 64 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 18 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 18;

// HELPER FUNCTIONS
// ================================================================================================

/// Applies the Anemoi S-Box on the current
/// hash state elements.
#[inline(always)]
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

/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
#[inline(always)]
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    state[0] += mul_by_generator(&state[1]);
    state[1] += mul_by_generator(&state[0]);
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
    state[0] += round_constants::C[step % NUM_HASH_ROUNDS];
    state[1] += round_constants::D[step % NUM_HASH_ROUNDS];

    apply_mds(state);
    apply_sbox(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_naive_mds(state: &mut [Felt; STATE_WIDTH]) {
        let mut result = [Felt::zero(); STATE_WIDTH];
        for (i, r) in result.iter_mut().enumerate().take(STATE_WIDTH) {
            for (j, s) in state.iter().enumerate().take(STATE_WIDTH) {
                *r += mds::MDS[i * STATE_WIDTH + j] * *s;
            }
        }

        state.copy_from_slice(&result);
    }

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
                    0x07e9a6981bc64437,
                    0xff6ab04ee232a314,
                    0x91f2d1e473c39b02,
                    0x021dfd98a0a69eb2,
                ])),
                Felt::new(BigInteger256([
                    0xa6ab21c48f401e51,
                    0x4b40acdab7589ecd,
                    0x956440dbd76387f0,
                    0x06f370939edd5561,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf02b90f33fe960d0,
                    0x9c97b25e4c7cc2b5,
                    0xca7c097c1c1f7497,
                    0x0b08197fd576ffac,
                ])),
                Felt::new(BigInteger256([
                    0xa06e4bbd846dbae7,
                    0x63fd128907853647,
                    0x7cd5bea332ea8dc2,
                    0x0ca1eee91a060083,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf550c2a2bbe3b67f,
                    0x27ae4eb41ad0207d,
                    0xf06fd31cd32ca1e8,
                    0x07f4cd98926a9e94,
                ])),
                Felt::new(BigInteger256([
                    0x83880fda945552d0,
                    0xe467226eca5a9392,
                    0x3f7d452a76a8dd1c,
                    0x10ec7b35f9d3f2bd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe241c92d63ba17cf,
                    0xa38f902f21887eeb,
                    0x32e2d05a6e53a407,
                    0x0e494be3d6aea2be,
                ])),
                Felt::new(BigInteger256([
                    0xb3338ae1cc24e0a3,
                    0xd4fa55aa5aee890d,
                    0x987d56b34dc2a06f,
                    0x0eaa6878d75749c3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0bda9740913c22f7,
                    0x105ce31fca8248bc,
                    0x9d32da44eb944201,
                    0x06647632051af9a3,
                ])),
                Felt::new(BigInteger256([
                    0x5ac7ceba7a0c8d91,
                    0xbf5b01b784b13f3f,
                    0xbae4f764ec12771e,
                    0x09dc9e08ec4d6ea9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7954da45193d45d3,
                    0xea3ed9505750dcef,
                    0x35acf18b0eb4bd79,
                    0x0caf7de01cc5cc64,
                ])),
                Felt::new(BigInteger256([
                    0x365cbdb6ccdddd4f,
                    0x9755977fd462bd9a,
                    0x7023b3a44fa51013,
                    0x08737a611d65e1fc,
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
                    0x8cf500000000000e,
                    0xe75281ef6000000e,
                    0x49dc37a90b0ba012,
                    0x055f8b2c6e710ab9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe9bd80afd48334f6,
                    0xb08ba3f89193de0e,
                    0x60fe916bb1314021,
                    0x0fe6ede65d5573a9,
                ])),
                Felt::new(BigInteger256([
                    0xbba99b953a5166c8,
                    0xe0c5bb74fb426643,
                    0x11c8e3721280441c,
                    0x0269bce008e98fdd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3ca8ffaf2fd56321,
                    0xa7008902000f08a4,
                    0xba20d120328112ff,
                    0x0bbedf4d8ed9cdf8,
                ])),
                Felt::new(BigInteger256([
                    0xf8c4355236fe86a8,
                    0x61c522c174ba35b8,
                    0xe4077f36d3d56a6c,
                    0x0babda3faa160297,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x872c9e78bff7b407,
                    0x26bb3f9d26e33d63,
                    0xb5b8ddb6c68ece27,
                    0x0d80fe3865b31dcb,
                ])),
                Felt::new(BigInteger256([
                    0x38d6d325d31ed930,
                    0xabd22218b74067a7,
                    0x03a23ed3137b3e99,
                    0x056998182c3a8de6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8e5ea733820f52cd,
                    0x180a4144235f023e,
                    0xa686d3184f7c6374,
                    0x10c2779ca91a6591,
                ])),
                Felt::new(BigInteger256([
                    0xf83fed1a114fd80f,
                    0x8964eba4c978b00b,
                    0xe931662da7827e95,
                    0x0fc72cf4c2c5fcfa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x664e9d2678aa9b16,
                    0x0243b4525b2bfd9f,
                    0xd05f7b45df65708c,
                    0x02a79b6f39b223f6,
                ])),
                Felt::new(BigInteger256([
                    0x5283bedc9df8caa5,
                    0x5e93e5c6fcdcb3c7,
                    0x50d46dc67507ec46,
                    0x04e263cbcbacb04b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd3035e1ceff53cbd,
                    0x157f7343b128ae12,
                    0xf53089f6ce8a2f66,
                    0x0c33f64ddb82c89c,
                ])),
                Felt::new(BigInteger256([
                    0xeecf610e06d78e50,
                    0x72b392a91ad47573,
                    0xd4a35cd8c0baa80a,
                    0x01e17f923a27ee34,
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
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0xf8ad709e68c2dabf,
                    0xa1b5dbdb51c1aa2b,
                    0x2b448406c4624c79,
                    0x08bfb020d8b9a8e4,
                ])),
                Felt::new(BigInteger256([
                    0x9b1aebca216d0ed4,
                    0x75d85b97a449d9c4,
                    0x4253cf99807a0e3a,
                    0x10bb8ac733ea0ecf,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x22373c9cf921467d,
                    0x295505fef66f4798,
                    0xbf7c701d8dc1265f,
                    0x047e8e2ea1f76e07,
                ])),
                Felt::new(BigInteger256([
                    0xd48c08371c757a5f,
                    0x766f474af6563bc3,
                    0xee27c7c34aae40aa,
                    0x0394eacc79d440c7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9af92643672ff26d,
                    0xdca9a3844e34d859,
                    0xcd17cf642248b34e,
                    0x0ddf76e710b423ea,
                ])),
                Felt::new(BigInteger256([
                    0x290f707147c6e44c,
                    0xb2af83343c4afeed,
                    0x6a8d2c9c2490c318,
                    0x0acbf62c6bb360bf,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x62e758b689e98371,
                    0x41e2e60ad8c6a6d8,
                    0x4910de00950d0afa,
                    0x099ed572389cc6c0,
                ])),
                Felt::new(BigInteger256([
                    0xcff27ce92a33f3f0,
                    0x9e2578d49fd2de44,
                    0x424aa71c14977f94,
                    0x053d6d1e992dfeeb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd7cb0cd40f6c44d7,
                    0x1548b0ea71ddad82,
                    0xd7aceb19c84ba7e4,
                    0x0269ef2840e2d9b2,
                ])),
                Felt::new(BigInteger256([
                    0x7fffbdf9249d91f5,
                    0x7520833986945b43,
                    0xcd7907d8a76d55bf,
                    0x0bc0fd7b36454b3f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfdfd98ce8a9f23ec,
                    0xa16f054120c0cae6,
                    0xaac144f4534b5f9b,
                    0x09cedfb69cb083aa,
                ])),
                Felt::new(BigInteger256([
                    0x50ad10204986b84d,
                    0xa623c68e167dd6b0,
                    0x6a10453ebc744861,
                    0x0772e3f9cc5db960,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x9c777ffffffffec5,
                    0xab3f94760ffffeb8,
                    0x02251ba4877a6e56,
                    0x071a44984b108eb7,
                ])),
                Felt::new(BigInteger256([
                    0x94c3ffffffffe4d8,
                    0x02d0883f7fffe3c6,
                    0xdfb1bf87b7bc5b55,
                    0x01872ef533960e4d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x296c7ffffffffed3,
                    0x929216656ffffec7,
                    0x4c01534d92860e69,
                    0x0c79cfc4b9819970,
                ])),
                Felt::new(BigInteger256([
                    0x7568ffffffffe606,
                    0xc9e8e8d8dfffe500,
                    0xf464b958816dfcec,
                    0x07b8c48f14411a33,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0x296c7ffffffffed3,
                    0x929216656ffffec7,
                    0x4c01534d92860e69,
                    0x0c79cfc4b9819970,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x839fb3fd482220e3,
                    0xc0fc6efb301a60fc,
                    0x506254d89a85c564,
                    0x0379b1db435a03f4,
                ])),
                Felt::new(BigInteger256([
                    0xc28e638e545be252,
                    0xa4e00932868e2f73,
                    0xa7f5e5bd571a44dd,
                    0x12833e2494f3d074,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3df9f1596b39caa3,
                    0xf03b4a74e1d86a68,
                    0xb416666c87dbf4ff,
                    0x089d2646b18269dc,
                ])),
                Felt::new(BigInteger256([
                    0xc357c5e6536ce457,
                    0x9adf07623eef60ae,
                    0xa10991e75d686e9b,
                    0x06643f2db54ae45f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9f694fff924790e8,
                    0xaa16df10eca6c0ad,
                    0x0c11ba4497e5875c,
                    0x08b278ea7dda0efb,
                ])),
                Felt::new(BigInteger256([
                    0x776d5067d9ed5832,
                    0xcffe08b4729f8dce,
                    0xad082b51981d8501,
                    0x0f7064a136b234f2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x055514c02a607a0b,
                    0xbf1c7e57b4e5c0bc,
                    0xb7416bb430c3e1b7,
                    0x0ce1d5dbc584cef1,
                ])),
                Felt::new(BigInteger256([
                    0xae3bc56cce7e70d3,
                    0xc99b5b6ffb916e5d,
                    0x575963d0de29954a,
                    0x089add7689fa17a1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x54e1e03d34f6cfd8,
                    0x986beceb729d8542,
                    0x96ebad2d7cdd1646,
                    0x124b91f11692ebcb,
                ])),
                Felt::new(BigInteger256([
                    0xede8033bb1d36e6f,
                    0xd9c0a58f801dcedf,
                    0x763b492575a41fb2,
                    0x0384d211e70f5952,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x923f7b94dc32fa81,
                    0xc183e681bf913e04,
                    0x61d08247475467f3,
                    0x05a7e6dac12ca1ea,
                ])),
                Felt::new(BigInteger256([
                    0x9ba72eeb35e83f5c,
                    0xd3d054bcdcf92b0d,
                    0x2d0d5b8a582f674a,
                    0x0130f5302efb1e28,
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
