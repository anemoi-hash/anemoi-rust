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
        *t -= y2.double() + y2
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
        *t += y2.double() + y2 + sbox::DELTA
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    let xy: [Felt; NUM_COLUMNS + 1] = [state[0], state[1]];

    state[0] = xy[0] + xy[1] + xy[1].double();
    state[1] = xy[0] + (xy[0] + (xy[1] + xy[1].double().double())).double();
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
                    0x736a2ba6eb472681,
                    0x2a166b08c7b34d15,
                    0x3beb22ea07004af8,
                    0x21d350da4df3e6db,
                ])),
                Felt::new(BigInteger256([
                    0x321f391e9828c101,
                    0xf65d49033acd9b6e,
                    0x2e94f3623fcbca2a,
                    0x2c7a67ff364d90cd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x78daf6827e67c2e1,
                    0x590dc6ba603ef856,
                    0x1047e716d31198da,
                    0x295fac6cc592ecd8,
                ])),
                Felt::new(BigInteger256([
                    0x9447697fd268a98c,
                    0x264922d7c02d45dc,
                    0x45db8f7bb8c1f58b,
                    0x1e426d92f9533bf2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x435dc40ae5608c21,
                    0x061dfaaf3fbda1c8,
                    0x15ab5abab89e6041,
                    0x1851b30bb3f28d9c,
                ])),
                Felt::new(BigInteger256([
                    0x8e77feb5acc392f8,
                    0x5ce3156eca59594f,
                    0x0319a80a9b09107b,
                    0x0803cebebc08934c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x484ff97129a11642,
                    0x16896d115949fded,
                    0x74d7c5ef9bea92ed,
                    0x2f4254d7d22ff215,
                ])),
                Felt::new(BigInteger256([
                    0x6a67dc05e0046e45,
                    0xc0be760c6899b5f0,
                    0x8c38cfa9506f4db6,
                    0x02cab7622633a7ba,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x38e3b4ac201adeb3,
                    0xb008f7bc976247ca,
                    0x3d365e151c10fb12,
                    0x26d0d0bb6540ebf8,
                ])),
                Felt::new(BigInteger256([
                    0xf4fad739920f606e,
                    0x62b040b89b37e401,
                    0xfb2cfd143f87a967,
                    0x029aeffd01bb15ee,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x796c3c2845ec1d50,
                    0xb708f56f879485ed,
                    0x7c2ed6bdc50849a5,
                    0x1d1fc1a4f8f8cec5,
                ])),
                Felt::new(BigInteger256([
                    0x2172c8a72d82c408,
                    0x4fdaa260f5515400,
                    0x310b83951100f580,
                    0x184f22beb6bc2f6a,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0xafd49a8c34aeae4c,
                    0xe0a8c73e1f684743,
                    0xb4ea4db753538a2d,
                    0x14cf9766d3bdd51d,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xbebb490c2e6f962f,
                    0x465ac2ad04e364b4,
                    0xff4b8beb7ecbd2f7,
                    0x1daa245477c958e5,
                ])),
                Felt::new(BigInteger256([
                    0xe07a0c33042f6fc0,
                    0xfbc9e1489023576a,
                    0xd40daa7093c5a810,
                    0x01eb281368a433ca,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9221c13bf351088f,
                    0x7ba14974192568d6,
                    0x764d83e12590a30c,
                    0x1efdfa43167d3d99,
                ])),
                Felt::new(BigInteger256([
                    0xa8fad3571dcabf05,
                    0x20a54df8cdb610bd,
                    0xb562a9b2221bf6a7,
                    0x113c1d95ad2f3ee3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc1291cac726de779,
                    0x730b09508e12a9ad,
                    0x965495beb3b74a80,
                    0x1c9527fa5aabb1b1,
                ])),
                Felt::new(BigInteger256([
                    0x68c3488912edefaa,
                    0x8d087f6872aabf4f,
                    0x51e1a24709081231,
                    0x2259d6b14729c0fa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x11e6e9777ce13053,
                    0x80cecce213130915,
                    0x9339feb9da4a476c,
                    0x00099f3ad0e08a3a,
                ])),
                Felt::new(BigInteger256([
                    0xca690fc49a5eefa6,
                    0xd8a291c2cfc8aca7,
                    0x61a01f3a73a1ce43,
                    0x0aa9fa9f93bbc9f4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x71968c81176cf29b,
                    0x3e98068063327f23,
                    0x17d5a77818b20498,
                    0x16bb81d796c4b5e6,
                ])),
                Felt::new(BigInteger256([
                    0x472d3428408515a1,
                    0xe3e52714073c437f,
                    0xa26b4b6fd09846f5,
                    0x0c859ebf44c3b39d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd45745e5fa3bf04d,
                    0x5a3563de9a4ed039,
                    0x7b3a3140af0b17a8,
                    0x28aaee78bc798d14,
                ])),
                Felt::new(BigInteger256([
                    0x267a3e11d0c88f62,
                    0x531634669ad908a9,
                    0x4d3ae25b1d053b5b,
                    0x254b78419d1b8f86,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xebfe324e72d60b25,
                    0x67b3d035045707f7,
                    0x287473d5fe75d745,
                    0x15f0f53d17dadd32,
                ])),
                Felt::new(BigInteger256([
                    0x20e2f3bea9d064e2,
                    0x13115e78480cfaa5,
                    0xe475b4c122bccbc9,
                    0x13b5309d4990e82c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf4d271d69ea20eb0,
                    0x15202d6ba355fa0f,
                    0x7fa7cf2152340e2a,
                    0x21f687ad6cf1fad4,
                ])),
                Felt::new(BigInteger256([
                    0xed9db25b732fca0e,
                    0xbd09fd9c48c26c50,
                    0x37c93ca75bca1c54,
                    0x1eeae136fbf06017,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1e3d9b573da8b00f,
                    0xe785d5c730191f4e,
                    0x229480718cfc1f41,
                    0x30241f3c73b0ea78,
                ])),
                Felt::new(BigInteger256([
                    0x77a4771a6f34d3e7,
                    0x7812ed0196ab2b8f,
                    0xf80db503a9432ff4,
                    0x2940faa831e6a02d,
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
                    0xd458f6ec603e7fe1,
                    0xbda8f9f8f482a527,
                    0x434d1fc7e2dfd33d,
                    0x12a6b65d0679c8b7,
                ])),
                Felt::new(BigInteger256([
                    0xc57fa9d73c9d752e,
                    0xecf30095cb23b31b,
                    0x00defec34539a323,
                    0x288b7c6fc0efcfbc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd0027680d946e249,
                    0x17c670bb5019041c,
                    0xe7c7e0664b64a41a,
                    0x11dc053fccb82f90,
                ])),
                Felt::new(BigInteger256([
                    0xa0702c3c078f4c53,
                    0x819b7a5513a8cb8d,
                    0x22cab1300183a5e7,
                    0x25629b5cfd0e5c83,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1237676824327534,
                    0x5668ff1e060ea509,
                    0x97b68d84fe3732b2,
                    0x28edace3412bad1b,
                ])),
                Felt::new(BigInteger256([
                    0xd10a085fd34b8d07,
                    0x23be267fad4056ff,
                    0x81fb19cc4da6c481,
                    0x0fc5d8f2bf85ebdc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf04d2159bae3d385,
                    0xad25aa67cb31ce35,
                    0x73aaefa7ad7185d4,
                    0x02cd99c4159956c1,
                ])),
                Felt::new(BigInteger256([
                    0xf3fc43c4b8af8292,
                    0x875bcdee87cd6166,
                    0xfb5f419e6e859f4b,
                    0x04a07c164f210c63,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x28fab36b4653dcbc,
                    0xca02f1e79cadaa9b,
                    0x9a092497977dd31a,
                    0x0bd522065e1ab221,
                ])),
                Felt::new(BigInteger256([
                    0x0a1bec7a0bf79729,
                    0x3c0aeefc57802994,
                    0xe36e4e6993bdb995,
                    0x27cb49126ef93e7b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd325d9250731ec66,
                    0xcda26b115086332c,
                    0xfffd641a36cef2b9,
                    0x044dd814119343dc,
                ])),
                Felt::new(BigInteger256([
                    0xd1910c2e11405972,
                    0xa0c277b2259b6254,
                    0x7e61492d86410b1d,
                    0x037199ddd9cb6d3f,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x115482203dbf392d,
                    0x926242126eaa626a,
                    0xe16a48076063c052,
                    0x07c5909386eddc93,
                ])),
                Felt::new(BigInteger256([
                    0x075ac9ee7eccb924,
                    0xc19fb16041c6327c,
                    0x0aad7b8599a48723,
                    0x255b297c2ed174eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7a17caa950ad28d7,
                    0x1f6ac17ae15521b9,
                    0x334bea4e696bd284,
                    0x2a1f6744ce179d8e,
                ])),
                Felt::new(BigInteger256([
                    0xc9638b5c069c8d94,
                    0x39b65a76c8e2db4f,
                    0x8fb1d6edb1ba0cfd,
                    0x2ba010aa41eb7786,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0x7a17caa950ad28d7,
                    0x1f6ac17ae15521b9,
                    0x334bea4e696bd284,
                    0x2a1f6744ce179d8e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xac96dc44651ce4dd,
                    0x557f2697850a2960,
                    0xd54990a4af8a0bee,
                    0x2b808ec686e5f797,
                ])),
                Felt::new(BigInteger256([
                    0x16e29a5fe27d2bf0,
                    0x26ec34a820eccf96,
                    0x57cadf8dcf53bdd6,
                    0x19e03d6ab20cd606,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3911e3073efaccb4,
                    0x6d960a97ba2fd1ab,
                    0xdf8768894cece515,
                    0x213b3a71018004c6,
                ])),
                Felt::new(BigInteger256([
                    0xd364bd241385b7e1,
                    0x9b5ac4f97154ab74,
                    0x50c05f5ee547a46c,
                    0x284badca3f2b2a84,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4934f470c5981f02,
                    0x2a22080ba55ddf7b,
                    0x6557953365aa27d8,
                    0x27dae9489e8bd087,
                ])),
                Felt::new(BigInteger256([
                    0x3467cd847319ef7f,
                    0x7321697fcc766057,
                    0x41614df97ba28b4e,
                    0x268df7e6d8c61d1f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcc41eca7e4f25b3b,
                    0x433914336299f26a,
                    0x65c8b482f90263b7,
                    0x10af0e0702fc7bed,
                ])),
                Felt::new(BigInteger256([
                    0x1ca17da58f0996fc,
                    0xb9859ff747296e1a,
                    0x74691970d80b7213,
                    0x064957b876e4e002,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcf0d60abb940a7a9,
                    0x4f20e9b9d24a923c,
                    0xd3b384674fb44f1f,
                    0x226e6057e8a32d41,
                ])),
                Felt::new(BigInteger256([
                    0xff02f64f86bf9396,
                    0xfa6ad706fd7c4b2f,
                    0xede850327fd7f637,
                    0x2e4dcd34667f85ed,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x47d8fdaf3af2f8bc,
                    0xafe9d227c1585a2b,
                    0x7b213fa2c9921412,
                    0x0ea2a5ad9ef58b9b,
                ])),
                Felt::new(BigInteger256([
                    0xa91c053bc21943a6,
                    0xb07fee2969a470d6,
                    0xefc50815e2f74755,
                    0x2f598ae6b6ac1011,
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
