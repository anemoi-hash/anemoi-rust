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

/// Two elements (64-bytes) is returned as digest.
// This is necessary to ensure 256 bits security.
pub const DIGEST_SIZE: usize = 2;

/// The number of rounds is set to 35 to provide 256-bit security level.
pub const NUM_HASH_ROUNDS: usize = 35;

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
        let beta_y2 = y2 + (y2 + y2.double()).double();
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
        let beta_y2 = y2 + (y2 + y2.double()).double();
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
                    0xac8a1dd37951cb2e,
                    0xadcc0d9b4404b243,
                    0x494ae73510611d55,
                    0x2022c9d22795ab9d,
                ])),
                Felt::new(BigInteger256([
                    0x1f4f22d497da12c0,
                    0x6bbb3ad6c87207b5,
                    0x7eeaa5e34a91a74d,
                    0x47ad5a0074806a75,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa4a30809fce2ed0f,
                    0xcc19f6769f6f7c65,
                    0xd276e3059434efea,
                    0x180f8bab0a896a59,
                ])),
                Felt::new(BigInteger256([
                    0x523f5da34b649e8d,
                    0xe468e7be0a546fe5,
                    0xd32ec9a813abb643,
                    0x45404b9c7aa3c96c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5f0788602ebebd73,
                    0x5c80a869fa6e6c9c,
                    0x5384032eb33b060c,
                    0x72fcef8df23412a9,
                ])),
                Felt::new(BigInteger256([
                    0x93224420dfd63424,
                    0xce1b300078b3ae09,
                    0x5c48536816e74d30,
                    0x02fe91ffa3507c3e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1e40516692816d92,
                    0x6c43f91313ebfca7,
                    0xca566413501b5760,
                    0x5fe23b15ab4723c1,
                ])),
                Felt::new(BigInteger256([
                    0xfdaa0a2331777000,
                    0xe70faf7352d64644,
                    0x170089068a8ec5b7,
                    0x18d7359445e15592,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb9ffa550862c4a9d,
                    0xc1326ac724f0f740,
                    0xde152f3e3bf87372,
                    0x0eb6cf46b2741c23,
                ])),
                Felt::new(BigInteger256([
                    0x8100cfab38a17cf4,
                    0x48ea4581094e7206,
                    0xf905b57ef7b91e02,
                    0x7227e373b5103a86,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x53916c54a25e8f53,
                    0x01c7411924e0bde9,
                    0xae04ee7fc411e510,
                    0x1ae14a6a6eabeb7d,
                ])),
                Felt::new(BigInteger256([
                    0x6bf380fb6eece556,
                    0xec45ed8e30891a36,
                    0x60fd78839ce79982,
                    0x0cb3af59a8397456,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0xdb6db6dadb6db6dc,
                    0xe6b5824adb6cc6da,
                    0xf8b356e005810db9,
                    0x66d0f1e660ec4796,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xd99f4a8fe00e11bd,
                    0xe12a753d82b33b30,
                    0xba87bf86c47e6f0b,
                    0x58de312b5993b67f,
                ])),
                Felt::new(BigInteger256([
                    0xb4c1972e044567f4,
                    0x8e14aacbecb9fb59,
                    0x2c0f182612f81cd9,
                    0x34d6a5db27797543,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x34413b9512617cf9,
                    0x7e6bdaab3f8443c5,
                    0x752178862c9a6ab7,
                    0x3dde7b14d2951772,
                ])),
                Felt::new(BigInteger256([
                    0xbdf2fa0b931e7a04,
                    0x60f39c6eae7a9a3c,
                    0x34e97cf1bd62a467,
                    0x13b9e8a487cc4745,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdb6db6ecdb6db6ca,
                    0x035ffa14db8a4eec,
                    0x5ea2264f581fdd5a,
                    0x401b2e0d73d97883,
                ])),
                Felt::new(BigInteger256([
                    0xfffffffd00000003,
                    0xfb38ec08fffb13fc,
                    0x99ad88181ce5880f,
                    0x5bc8f5f97cd877d8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1d9e7eb26e2b7f30,
                    0xc29c67043c66534d,
                    0xea0f5bb48f55d399,
                    0x54a23b8eb5829c7e,
                ])),
                Felt::new(BigInteger256([
                    0x8d0a1b5d94fae5f1,
                    0x64fb05399d9c4dd8,
                    0x94071e3519ec530d,
                    0x30a9b41a9c63f2f9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x54ac6c0ff5457d4a,
                    0x5977f9d7d4d9f19e,
                    0x7b4894ad898dd4c2,
                    0x48aa067b0d785ede,
                ])),
                Felt::new(BigInteger256([
                    0xb28e074be851fb11,
                    0x1cbefafc9a433db8,
                    0x510f5045f270bad1,
                    0x259f30dce7d28aa2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1f46aa1bfe122e8d,
                    0x8361e8ed143ec6b4,
                    0x403b60e6367d54a1,
                    0x350c6937bd56b0ca,
                ])),
                Felt::new(BigInteger256([
                    0x4cf17c05afcce567,
                    0x4577ff4283355610,
                    0xa2842957d13fc98d,
                    0x371c8e6be97ae1a9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x95b359dcf9340017,
                    0x3d834f7af43bb19f,
                    0x3762dc59f4e0981b,
                    0x27b4bab7c44a57f3,
                ])),
                Felt::new(BigInteger256([
                    0xb94caed95e676617,
                    0xb6b4a1b0270541bc,
                    0xe8d2f7d2a7238787,
                    0x1c7cf34e89063fac,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa2e49599f7ecab65,
                    0x9331f57433d8bf7c,
                    0xb04b05d5e74a8213,
                    0x059a0ebd8ab8582d,
                ])),
                Felt::new(BigInteger256([
                    0x05c9717836f21f19,
                    0xe01a26f38424cb98,
                    0x900c043a52dfe1eb,
                    0x6fde8740f7a722d8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x707d0f8e77235d80,
                    0x3648cca2cd245aa9,
                    0xb44a86f8dddf28cb,
                    0x33ae301f34c1ccaa,
                ])),
                Felt::new(BigInteger256([
                    0xcd38f699bcb2b41b,
                    0xa80b91cfc444632c,
                    0x90d58b307e33bfc4,
                    0x37c1fd038e00658a,
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
                    0xc0ee52864b130be0,
                    0x5e2b7f3feb20735f,
                    0xc183d0e97123e99e,
                    0x1956a400bb5dc2e9,
                ])),
                Felt::new(BigInteger256([
                    0x0b266a20c24e6ac0,
                    0xa309bba0426b18fb,
                    0xf91bbef9a86129fd,
                    0x07f2d05b3787087e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8957577958ee837f,
                    0x267b5ed59fe0ba89,
                    0x0681cbe90b5a3b1f,
                    0x0ee8c65c2efa40c3,
                ])),
                Felt::new(BigInteger256([
                    0x2939dee8adf561a7,
                    0xe735ac58abea9779,
                    0x77b8bb6ee4ca5bdb,
                    0x6d9ef16eaa6a751d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb7fe9017a035c722,
                    0x2c98cd74d06ba4be,
                    0xe3a2063052c48406,
                    0x1d758157adb4cf89,
                ])),
                Felt::new(BigInteger256([
                    0x1b4eb44957a73dfb,
                    0xec9f8c077002c2e6,
                    0x281bcf5c98fce16f,
                    0x6e83a43f5d05f688,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdf83888252498790,
                    0xb96cc41f1baedecc,
                    0x054b107bf6f11d1c,
                    0x07ef52c3650272cf,
                ])),
                Felt::new(BigInteger256([
                    0xcda9b5ece099eb1d,
                    0x0a030291f02ff830,
                    0xc2d7fab20213fc8b,
                    0x2e485c44bcbd9a70,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x88c83e0d5083e36a,
                    0x63c0dbb28cf62b3c,
                    0x98ae194fdf17bac2,
                    0x5627cbac57f780ff,
                ])),
                Felt::new(BigInteger256([
                    0xc2495996a56b4f7f,
                    0x3e64b35c99cea3ea,
                    0xa5338d4098fc62f0,
                    0x6c3c0d47fbb946b3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x57a6a07186418e98,
                    0x9c1bcfb3b1c48af1,
                    0x8066faedf289d6b2,
                    0x65cb0a2205493bc1,
                ])),
                Felt::new(BigInteger256([
                    0x4596f285245fb7e1,
                    0x7f274d3ace06b0f6,
                    0x8a18ced2dac434a5,
                    0x4421a38c96e58165,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x00000010ffffffef,
                    0x70681bcd001be411,
                    0x9928a7775c40a7a5,
                    0x4d37e37a3c8aae34,
                ])),
                Felt::new(BigInteger256([
                    0x0000007cffffff83,
                    0x1c66ea8900cd147d,
                    0xfcc184134bf98566,
                    0x64f54c64ae19d3be,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0000000efffffff1,
                    0x17e363d300189c0f,
                    0xff9c57876f8457b0,
                    0x351332208fc5a8c4,
                ])),
                Felt::new(BigInteger256([
                    0x0000006dffffff92,
                    0x048386b600b4786e,
                    0xfd252c8bdc752db6,
                    0x2fe21a441e542af9,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0x0000000efffffff1,
                    0x17e363d300189c0f,
                    0xff9c57876f8457b0,
                    0x351332208fc5a8c4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0efb396b9b37f720,
                    0xd36fa0a1bc0e223d,
                    0x914609bd0bcc0f8d,
                    0x50fa567f400efe62,
                ])),
                Felt::new(BigInteger256([
                    0x7404fc1600d62c9c,
                    0x1c20900066d498aa,
                    0x251ea304d46e36c9,
                    0x6f149089517a0810,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa9ec6fdc1aa42f0a,
                    0x82813d305354b6df,
                    0x1933dbc11317ae02,
                    0x56af746fde3284e0,
                ])),
                Felt::new(BigInteger256([
                    0xceb0edf36872aae7,
                    0x824c8098f3456f9c,
                    0xf4c8ad8630a50dcd,
                    0x14d9348ac41b278c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x77257e1f05c878f9,
                    0xae83c996e088d10f,
                    0xc909a18847e39df5,
                    0x6b7d13203f2d9d91,
                ])),
                Felt::new(BigInteger256([
                    0x5d55272980228cc3,
                    0x680a931293cbf659,
                    0x40ca51de4cc94b02,
                    0x336f96daf3f6d88b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7f2781fe767ef559,
                    0x58068e16ad01f024,
                    0xf2bf3b49f23954df,
                    0x640e89fe3af6b153,
                ])),
                Felt::new(BigInteger256([
                    0x47be43e81e12a086,
                    0x7bbf0d1eab476136,
                    0x32b7898767da3e86,
                    0x331c36455fcb840b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd8c9b132d6730fdc,
                    0xce524725c1a822ae,
                    0xb6820ddbca91872e,
                    0x204c955e16ba02ef,
                ])),
                Felt::new(BigInteger256([
                    0xafcd31fc8290be81,
                    0x3b295d5ee56adeb4,
                    0x3c4e3e330fb3652d,
                    0x6678d434479460b1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3ec7421984df95bb,
                    0xc7385c4353f9f1b1,
                    0x7a2d4291c75fe723,
                    0x72ffe5ad7f19d067,
                ])),
                Felt::new(BigInteger256([
                    0xfd09c13ec67ccff7,
                    0xa78256fd19e7c8d5,
                    0x7ac0b8970af69e7a,
                    0x3da15804ed4bc740,
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
