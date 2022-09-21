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

    let sum_coeffs = xy[0] + xy[1];
    state[0] = sum_coeffs + xy[1];
    state[1] = xy[1] + (sum_coeffs + xy[1]).double();
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
                    0x19307400a934e412,
                    0xfd3992176b96d503,
                    0xf8efffeae8782538,
                    0x72fcbba45f6dd0dd,
                ])),
                Felt::new(BigInteger256([
                    0xadec1ac6afb3ee89,
                    0x35e40065edabb8fb,
                    0x7bb7876d403cd76a,
                    0x659b4a7550a9607d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8d30c3f3a5e50e40,
                    0x552eda36975fe462,
                    0xe7469fcf263184db,
                    0x679dbfd97c11b604,
                ])),
                Felt::new(BigInteger256([
                    0xa5f99ddfb64a8941,
                    0xa4dd1afef98ddb15,
                    0xbd795f253e56ad3e,
                    0x06fc19331e86cb9e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x218519305b901194,
                    0x34af26057b012d66,
                    0x0fc17fd665a7151f,
                    0x50812695d0cc45ff,
                ])),
                Felt::new(BigInteger256([
                    0x2ab7616061c48a3e,
                    0xa813d654dcd2e389,
                    0x2f430fb36939d4f1,
                    0x10f6d1e7c20e3b51,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa56dcbbf8e1a2a2d,
                    0xb37082614d397472,
                    0x4bd34a578dd58997,
                    0x4e7dde2c955f1975,
                ])),
                Felt::new(BigInteger256([
                    0xcd0c1d747650460a,
                    0x11d13816b99b85b5,
                    0x4d08177228f0c911,
                    0x331518347ca55013,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd4be8a654dff7e13,
                    0xcf9211d920ca7d7f,
                    0x65b5c6104f3e1c36,
                    0x0be146f01a982b9c,
                ])),
                Felt::new(BigInteger256([
                    0xc8c807ec23d1eba9,
                    0x181a5b2a6f1d677c,
                    0x22924dadb984ea3e,
                    0x73b8642aeb4b9713,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x341366edfb855c7b,
                    0xe06d57cb25208481,
                    0xf167939b58d4d8c0,
                    0x46546a7218bd1619,
                ])),
                Felt::new(BigInteger256([
                    0x75531c0c3d1b44ae,
                    0x97e84e2d99ce8b79,
                    0x5bc216c22762f93a,
                    0x5de3f079377f1131,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x00000005fffffffa,
                    0x098e27ee0009d806,
                    0xcca4efcfc634efe0,
                    0x486e140d064f104e,
                ])),
                Felt::new(BigInteger256([
                    0x0000000efffffff1,
                    0x17e363d300189c0f,
                    0xff9c57876f8457b0,
                    0x351332208fc5a8c4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x00000003fffffffc,
                    0xb1096ff400069004,
                    0x33189fdfd9789fea,
                    0x304962b3598a0adf,
                ])),
                Felt::new(BigInteger256([
                    0x0000000afffffff5,
                    0x66d9f3df00120c0b,
                    0xcc83b7a7960bb7c5,
                    0x04c9cf6d363b9de5,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0x00000003fffffffc,
                    0xb1096ff400069004,
                    0x33189fdfd9789fea,
                    0x304962b3598a0adf,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7508a990089cc122,
                    0xc1864add46f18efc,
                    0x89eb5eb555ae2402,
                    0x565801e8ad859748,
                ])),
                Felt::new(BigInteger256([
                    0x97fd6de8c0ed70cb,
                    0x11754e1a7b921ef6,
                    0x291a94c7d8556f65,
                    0x2a6fffa05879947e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd923ffb4127a20c1,
                    0x4b2b6c318a7d3e8e,
                    0x2eff8611993d0753,
                    0x01a84aec8f81cffa,
                ])),
                Felt::new(BigInteger256([
                    0x58419d47db3ecac3,
                    0x3b33f3620e885833,
                    0x1b786b4870d0bbe5,
                    0x0a4caf0c3d8a6b93,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x76f3dbf11f192610,
                    0x84d6d2af34a6f478,
                    0x6e479f3d381abf02,
                    0x726eca6554e8bca1,
                ])),
                Felt::new(BigInteger256([
                    0x189f19449ff6d65c,
                    0x0a4633ad4624147c,
                    0xa55e9e1dc62ba2ec,
                    0x0df9180c18a4ba03,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3f8606a97abab640,
                    0x83554e8bc07223df,
                    0xb2a9a133d61543b4,
                    0x40ba6742650c3c53,
                ])),
                Felt::new(BigInteger256([
                    0x4c182ac86bc5b289,
                    0xc4be312b3a817175,
                    0x7f2181d1cb797874,
                    0x409c3f661d204b72,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x664e9a3f95a35563,
                    0x584b8027ff08947b,
                    0x4466b15baf0440a8,
                    0x0b76c09f9df45f32,
                ])),
                Felt::new(BigInteger256([
                    0x95653c6c4f18966e,
                    0x74f3b7776d303474,
                    0x7825d85d0deb9389,
                    0x16b83e16fd96d82f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1eb99f0875bbe5d5,
                    0x68c2ac2058c0e376,
                    0x4278110f94571b2b,
                    0x1a40fcbe34803dec,
                ])),
                Felt::new(BigInteger256([
                    0xb2c65a1e28931057,
                    0x15b0026b4b51f666,
                    0xad7860d9466f578c,
                    0x1e7842a276e20fc1,
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
