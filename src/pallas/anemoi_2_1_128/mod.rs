use super::{sbox, BigInteger256, Felt};
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

/// The number of rounds is set to 19 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 19;

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
        let beta_y2 = y2 + y2.double().double();
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
        let beta_y2 = y2 + y2.double().double();
        *t += beta_y2 + sbox::DELTA;
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    let xy: [Felt; NUM_COLUMNS + 1] = [state[0], state[1]];

    let tmp = xy[1] * mds::MDS[1];
    state[0] = xy[0] + tmp;
    state[1] = (tmp + xy[0]) * mds::MDS[1] + xy[1];
}

// ANEMOI PERMUTATION
// ================================================================================================

/// Applies Anemoi permutation to the provided state.
#[inline(always)]
#[unroll_for_loops]
pub(crate) fn apply_permutation(state: &mut [Felt; STATE_WIDTH]) {
    for i in 0..NUM_HASH_ROUNDS {
        apply_round(state, i);
    }

    apply_mds(state)
}

/// Anemoi round function;
/// implementation based on algorithm 3 of <https://eprint.iacr.org/2020/1143.pdf>
#[inline(always)]
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
                *r += *s * mds::MDS[i * STATE_WIDTH + j];
            }
        }

        state.copy_from_slice(&result);
    }

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x549cb1b12215b5f8,
                    0xb9320b7623080e0a,
                    0xe21f92850bed29b5,
                    0x00e7b83bf28e88f3,
                ])),
                Felt::new(BigInteger256([
                    0xc46849152c8bfa9e,
                    0x271a9e7cd4417d5c,
                    0xcf6c148b5c10c200,
                    0x3549b83894f08391,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x925f58c06480019d,
                    0x853a386fa7586ab2,
                    0x5f61c869a872aa48,
                    0x2c5a820518063f94,
                ])),
                Felt::new(BigInteger256([
                    0xf656c806584ea0d6,
                    0x5d7992593311d429,
                    0xd592dacf9b683056,
                    0x3d0d7d18565e34e5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xaac25d6464647e5b,
                    0xd56f947ac4734b24,
                    0xd94de15d25bb2820,
                    0x24be138a097e25db,
                ])),
                Felt::new(BigInteger256([
                    0x0dcc53187ca26b4a,
                    0x0e11e5386c6854d4,
                    0x9b66013dc4fc5a71,
                    0x02f11432fa70dc89,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd82affcb320a6cf0,
                    0x99ba4e4af06d5824,
                    0x847b0a3c9b874593,
                    0x2b8a60896dbedfc6,
                ])),
                Felt::new(BigInteger256([
                    0x4513d7dc197ec40a,
                    0xdbfdaafb008f2527,
                    0x37fed91c601c809e,
                    0x0513408c3ab1f33c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0949782eb779d666,
                    0x06c9ebfb395c1b7b,
                    0xbfb48908817e089d,
                    0x270579da9f12bb55,
                ])),
                Felt::new(BigInteger256([
                    0x691e725ea3f8a0a6,
                    0x1ca4640964701e25,
                    0x9db9d4a52f2a1073,
                    0x15a102922fd2eb46,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x880576e13cc90b1f,
                    0x68e9be46608cbe1f,
                    0x21699f5b3006d2aa,
                    0x3a4cc129c56b6261,
                ])),
                Felt::new(BigInteger256([
                    0xe6c770f10c7cc0c7,
                    0x0909c7d04855b38e,
                    0x9b3c7d1960ebdf46,
                    0x117da9beb0d61a5b,
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
                    0x64b4c3b400000004,
                    0x891a63f02533e46e,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x91036227f225de02,
                    0x55350fa37393df7a,
                    0x245fa221cf61c197,
                    0x123cf58f73eab51a,
                ])),
                Felt::new(BigInteger256([
                    0xd63d2cac361d05f1,
                    0xb7abe04b340581cb,
                    0x1e2782bab31ed630,
                    0x18a4ff9a32b13d3f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xac651fe78adb1249,
                    0xe9bd87b589846a1a,
                    0x854662239be6242e,
                    0x276fa8c10b1c97d6,
                ])),
                Felt::new(BigInteger256([
                    0xc192ab7fb075f32a,
                    0x8aad2735f4327278,
                    0xadb89774fbc2b737,
                    0x12662c0015bba883,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfdc9f7e0c891be47,
                    0x4a6f2de9d31dc82d,
                    0xa3579fd2dc392bba,
                    0x0afb7169ade45919,
                ])),
                Felt::new(BigInteger256([
                    0xf64184aa8e34742a,
                    0x13a822cc38a571f8,
                    0x354c482528f08ca8,
                    0x04fd9e24b40a1e11,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb47e1194546aed69,
                    0x41ad3e7d8a2304ea,
                    0x6bfb4210c3dd4910,
                    0x03161b862d3eb871,
                ])),
                Felt::new(BigInteger256([
                    0xb6e68fc09612048e,
                    0x1b0540f4cab4ed0c,
                    0x95c38b7e41232183,
                    0x3dfcafc421e4c0a6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdbfb9c57a4cf8f94,
                    0xb4f45f76d45c0011,
                    0xad4c8f76b5019584,
                    0x09da6dfe7ccdf2e7,
                ])),
                Felt::new(BigInteger256([
                    0xdd886bfb79a8652f,
                    0x5128fe3d83d7c981,
                    0x06748c0ef3cd7099,
                    0x382b675fa0fde2a4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf4ea799842a8cad4,
                    0x64e465740ceea336,
                    0x13bb704e5b3e577c,
                    0x2773b1b165690126,
                ])),
                Felt::new(BigInteger256([
                    0x04395e5734733d92,
                    0x708793e22b3563d0,
                    0x3cd0f208bad77d5a,
                    0x23141077d2678182,
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
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x4cacef84d85ccf3c,
                    0x47898be77a01770a,
                    0x62cdbf8a865628e9,
                    0x0422d289c37c56b2,
                ])),
                Felt::new(BigInteger256([
                    0x082665af0fd5c039,
                    0x89c18fae8d374f9f,
                    0x6ed7bf1aff1c00c5,
                    0x098c1c1af96e1f56,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc353a883f2750d99,
                    0xac5ad43a424d7137,
                    0xfc9d6f963fa216ad,
                    0x3b87c7b77b1d1c47,
                ])),
                Felt::new(BigInteger256([
                    0x7ef7fc2cbd0e3a5d,
                    0x45f22c9944336ad6,
                    0x99d27a150be038ae,
                    0x29e53e73130fe6ad,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe87904e5b6cdf668,
                    0x26d6bfe0dc01e429,
                    0xde3ff6b27649ee41,
                    0x171922901e599179,
                ])),
                Felt::new(BigInteger256([
                    0x9e95a558b4e6c27a,
                    0xae61ffc4510711d1,
                    0x49ebc20a0ff6d214,
                    0x049dbc9a840ab428,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdc7319840a017702,
                    0xac831d79b890d021,
                    0x1b401f186eae8313,
                    0x0e78b397339ab38a,
                ])),
                Felt::new(BigInteger256([
                    0xbfea8c0f86f85809,
                    0x07467b26386edd01,
                    0xa22e0b30de7f40d6,
                    0x174193dc4db0fb07,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2894707bf15b3957,
                    0x8629ae8547a07aee,
                    0x25b83dbb03b85593,
                    0x38987366fb1af3fc,
                ])),
                Felt::new(BigInteger256([
                    0x6852d4d410160557,
                    0xfd16e51901cacf6d,
                    0x4cdbd80d8716bcd2,
                    0x1191290a4923807c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdf1625e1f780bca2,
                    0x6d6dab20236c1f5b,
                    0xb9efe432f3be104d,
                    0x0e8b49de4d11e3ab,
                ])),
                Felt::new(BigInteger256([
                    0xab89efe42173363f,
                    0x9135eec8ec1cd0de,
                    0x7467e141a213572d,
                    0x33221ac56d1376fb,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x3cf09ab4ffffffe9,
                    0xeba8415b2a159e85,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x67497e20ffffff85,
                    0x88147ee788044fbd,
                    0xffffffffffffffef,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa1a55e68ffffffed,
                    0x74c2a54b4f4982f3,
                    0xfffffffffffffffd,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x5ed150a4ffffff99,
                    0x359872984207c5e5,
                    0xfffffffffffffff2,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0xa1a55e68ffffffed,
                    0x74c2a54b4f4982f3,
                    0xfffffffffffffffd,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x756cebf027899059,
                    0xf8515a503c160525,
                    0x8d047b1181e22cc4,
                    0x33df5f10a2a2f362,
                ])),
                Felt::new(BigInteger256([
                    0xee923dabd58591f2,
                    0xda3def4f947184eb,
                    0x2fee26728886e09d,
                    0x0ce8f76e269ce043,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd976d1afa3bc3166,
                    0x80fb4f48721aa2f9,
                    0xfdb9d1ff7b033214,
                    0x0d01fff6da6c9dab,
                ])),
                Felt::new(BigInteger256([
                    0x251ce3adefbb315a,
                    0xa8942007756ba09c,
                    0x8e73941272f03314,
                    0x2aef3e45572efb09,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x01653fa13f4fc2ca,
                    0x8ec0beb671253d42,
                    0x4fdac0e4c61c08a8,
                    0x2e2dd194b28f1643,
                ])),
                Felt::new(BigInteger256([
                    0xda0850b7f1759069,
                    0x1151ee606ada58c8,
                    0xd9318681ee82fd5f,
                    0x2b82d48200d62378,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x69ad73f7acdb2f2d,
                    0x8c565340c0212ef3,
                    0x4626570cc72ac741,
                    0x02c096e4b80f9ab0,
                ])),
                Felt::new(BigInteger256([
                    0xd04dcfe5e74043ea,
                    0xc4f61b69f914c7c2,
                    0x00edbe70c255251d,
                    0x25048653e5ff0079,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xffd836c641c95408,
                    0x330ef60a3dfc95d9,
                    0xa60375fea72a05b2,
                    0x106e409a68cc7669,
                ])),
                Felt::new(BigInteger256([
                    0xce5eb5c65904a97e,
                    0xda1b1a502e6cc393,
                    0x8aed2606cae8d94d,
                    0x23b86c0e5521d08c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd41311a29ec0cbd9,
                    0xba60f11c9ac84f46,
                    0xfff74a7b1e1ec430,
                    0x0e35cfb96e733694,
                ])),
                Felt::new(BigInteger256([
                    0x36bc17243b37317b,
                    0x12d40b5be8b96425,
                    0x743c55a938ad2c21,
                    0x3a2f2964955387e4,
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
