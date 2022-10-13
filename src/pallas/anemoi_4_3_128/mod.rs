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
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x3263957d45a582f5,
                    0x9811d8edb5d76596,
                    0xe7c4ff218fb2e0ad,
                    0x31fee62e20ee8cc8,
                ])),
                Felt::new(BigInteger256([
                    0xa482f8f57e3f9a09,
                    0x4330b1c95b1a9f15,
                    0xd6fa3dbfb5b30e23,
                    0x353a1859622d16ed,
                ])),
                Felt::new(BigInteger256([
                    0xb4ed60a5581a4fd5,
                    0xf4bdac3f53b4ef15,
                    0xa357254c9fa2dbf3,
                    0x3f9c122b98176962,
                ])),
                Felt::new(BigInteger256([
                    0x3eeeb90c8f23f687,
                    0xc3e67899e2938f94,
                    0x76896c2ff4a43ae4,
                    0x23072ac2a58269e2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x31035a0548abd251,
                    0xa25ec7d59322e797,
                    0x23b02f37a4d9d38b,
                    0x299fe02d86631e71,
                ])),
                Felt::new(BigInteger256([
                    0xfb9bf25282ee0c54,
                    0x35518a9a50e0a60e,
                    0xad90c203f2bd3ce9,
                    0x1f3698daa8af2ade,
                ])),
                Felt::new(BigInteger256([
                    0xcb0f40dd0ca1dbef,
                    0x6a34c8a81aef90ad,
                    0x6e4fed24452aff3e,
                    0x09c904129b40bd17,
                ])),
                Felt::new(BigInteger256([
                    0x033ce44e4225d357,
                    0x27bbffa25d76d432,
                    0xc72cc12886d53498,
                    0x380b55a4610be0c4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf0dabdaaab4ff56a,
                    0xcc6ab7354d435e48,
                    0x9bfa3469758b8cbd,
                    0x26e462522521dc53,
                ])),
                Felt::new(BigInteger256([
                    0xd4321b0f9f121399,
                    0x5be70868b0a6e39c,
                    0x48e56338b3e91098,
                    0x2814df9473a40b8f,
                ])),
                Felt::new(BigInteger256([
                    0x87a2b2f99f19f946,
                    0x776a115450815c3a,
                    0x526016c4fdefc0e3,
                    0x23fbfc6671c54634,
                ])),
                Felt::new(BigInteger256([
                    0x011c562e7591593f,
                    0xabc48ee6e372d3cb,
                    0xf8f5dee5046aae8f,
                    0x26cc3fddb0a98c0c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x33a2d602bd3ad18a,
                    0x46036cc3dfb7cd67,
                    0xfd82b6c253e66eb8,
                    0x1025d946e242ff26,
                ])),
                Felt::new(BigInteger256([
                    0xf602691d7f2556be,
                    0xd62fda7056958edb,
                    0x07f38b9196377e76,
                    0x05d36eaf008c858e,
                ])),
                Felt::new(BigInteger256([
                    0x1c8946b0f1b83781,
                    0x7cde336769c023a2,
                    0xcb965bc8e8731045,
                    0x0ee55b7b3ad5ccc6,
                ])),
                Felt::new(BigInteger256([
                    0xa2f38a83cb9e4bd7,
                    0x3702d5b0deac7688,
                    0x5d842f706aab049a,
                    0x247cc2c45eb6a74b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x45eaae106dcc48e4,
                    0x54fae907e3c4396d,
                    0x054be5e246e2d669,
                    0x2160e7ac9b2962fb,
                ])),
                Felt::new(BigInteger256([
                    0xc5293b044faf9475,
                    0x288ea64f00331321,
                    0x6d4a0203ab0ff3f4,
                    0x09dcb84a745c66b3,
                ])),
                Felt::new(BigInteger256([
                    0xc6c6fc6bdd141eee,
                    0x1706e6f1d938d9bf,
                    0x34a9cf942269063b,
                    0x20f1e492027e454b,
                ])),
                Felt::new(BigInteger256([
                    0xbe346cc99a8ac66f,
                    0x2a933342f227e99d,
                    0xf4bfaec129d5a184,
                    0x0843c4de0723747b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4a2e551fc959103b,
                    0x83a9dbfd3c0d1593,
                    0x9e15bf1c8b8a6336,
                    0x1d9c618b3b53c605,
                ])),
                Felt::new(BigInteger256([
                    0x680fe37aebb7b787,
                    0xe8add7a41e5fec83,
                    0x19b4c094e0e107e5,
                    0x2ed949154d19792b,
                ])),
                Felt::new(BigInteger256([
                    0x3fec301a2e0f8b68,
                    0xc08ab6314454a14a,
                    0xc9097f4a689fc541,
                    0x0db96dfae62050d1,
                ])),
                Felt::new(BigInteger256([
                    0x641d0475f5eb6efb,
                    0xc29dc93335cb141d,
                    0x460c7a1d29151597,
                    0x1ff486750a52cf6a,
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
                    0x3dbed2378e7866c3,
                    0xa3a216bbb8b9cf8d,
                    0x2aee4c31d40937d3,
                    0x074212f8b9cbe00a,
                ])),
                Felt::new(BigInteger256([
                    0x0599bb3199e63bd1,
                    0x10cf8d45a2fcadb7,
                    0xbd5a95eae9f1ad20,
                    0x05deaa6f24a7dfc3,
                ])),
                Felt::new(BigInteger256([
                    0xb3484385824e9667,
                    0x80c38d519a12e454,
                    0x89754b1e1971de93,
                    0x1bcaf54c4165ba52,
                ])),
                Felt::new(BigInteger256([
                    0x54cd90befe2f6f06,
                    0xe7206901bab65532,
                    0xdd3d868526681dbe,
                    0x13b8ebb23b90e2ee,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc75f6912310fa6cc,
                    0x859fd4d8350a2d92,
                    0x59e68340b984d5bc,
                    0x134406be76078092,
                ])),
                Felt::new(BigInteger256([
                    0xe4faddecbcfcbbcc,
                    0xebee15cb9a369dc8,
                    0x4fba74e5d49a21e1,
                    0x052615f778e9a58e,
                ])),
                Felt::new(BigInteger256([
                    0xe2298e95f307ad56,
                    0x39286c7fe34bd364,
                    0x1e304e6236fb1192,
                    0x13226f4a921db42e,
                ])),
                Felt::new(BigInteger256([
                    0x016ac9c244a58ccd,
                    0xd20817710d444513,
                    0x8798cc49e6ddfdfd,
                    0x126b9cba16aef130,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5ae7b65349774473,
                    0xb98b90a43143f934,
                    0x3d52e92428056488,
                    0x2724704576b0c7f5,
                ])),
                Felt::new(BigInteger256([
                    0x8ae0d456757134fb,
                    0xc934182c7be917a6,
                    0x9942c331d714fc38,
                    0x2ef29e4546ebeadf,
                ])),
                Felt::new(BigInteger256([
                    0xcc9ff08fb2e8c983,
                    0x4cd516123573a62e,
                    0x83c0063036334209,
                    0x293ab98ae35908b3,
                ])),
                Felt::new(BigInteger256([
                    0xc222ad3b8b4e2b4a,
                    0xe1e9dc8bb9f12443,
                    0x93e3f9a8055952e2,
                    0x3d839508cba01f11,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa31e8d4c77293c41,
                    0xaeab5b6f084499ab,
                    0x2aff0f8503eeaea4,
                    0x24a2dbb3164ae7d6,
                ])),
                Felt::new(BigInteger256([
                    0x501a9105ac3eb4ea,
                    0xee9ff65ab48f60ef,
                    0x1b34f05d80a5b814,
                    0x3792697e7cfd7588,
                ])),
                Felt::new(BigInteger256([
                    0x77b01f63fd0269b7,
                    0xa90929d753f99514,
                    0xff79d1437d23c7b6,
                    0x1da29d03b790779e,
                ])),
                Felt::new(BigInteger256([
                    0xcb638fc97b402a69,
                    0xf4b01a5153d7760a,
                    0xa769e3aa0b154def,
                    0x324d2a1eb216fe84,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x17a447d8b16540fc,
                    0xfe83e33a812bd2eb,
                    0x5ec94230b6171051,
                    0x078002cb77fa13e0,
                ])),
                Felt::new(BigInteger256([
                    0x2e62ffde7d2a8521,
                    0x59696261bdc48305,
                    0x7ec9c6d0805b55b9,
                    0x224db24a13f3a6c5,
                ])),
                Felt::new(BigInteger256([
                    0xfee046e8daf7390b,
                    0xbe64519091c01750,
                    0xb37133d358f50bc5,
                    0x31884ad7683e3ebd,
                ])),
                Felt::new(BigInteger256([
                    0x937b6b59e9d6f3ac,
                    0x011f8cf77c156234,
                    0x9eb688d190e0500a,
                    0x1148feace945f827,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb8d0427b61264477,
                    0xc5ab23e056810cc2,
                    0x73cb84e49606761d,
                    0x0586df58ed4256a8,
                ])),
                Felt::new(BigInteger256([
                    0x1d354ba13d7d7f51,
                    0xafdd5f211746ba92,
                    0x7b0458ec51868891,
                    0x16e296a686474a93,
                ])),
                Felt::new(BigInteger256([
                    0xf52aa079ccaa3924,
                    0x3c203cfc56e69edb,
                    0xf8b116154b50e935,
                    0x2a0176b2ac4c1ecd,
                ])),
                Felt::new(BigInteger256([
                    0xb9d10baebd922d09,
                    0x555cbbf2bbbd5da5,
                    0x031114c93d89991f,
                    0x1172a6ed3bbe299d,
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
            [Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0xca0c2362b29ce4aa,
                    0xf6bfecf93377e4a1,
                    0xce67935407abcb8a,
                    0x3ed8d8d63c5ae09f,
                ])),
                Felt::new(BigInteger256([
                    0xa525e0569b3bfaf6,
                    0x5a45d770c62ebaa3,
                    0x2bc3802cbcf650be,
                    0x01b077d81f2643b8,
                ])),
                Felt::new(BigInteger256([
                    0x540edb76e3d5363c,
                    0x398a4a1655f6fad4,
                    0x2c342d43eaf3a474,
                    0x02a7b932e082eeb5,
                ])),
                Felt::new(BigInteger256([
                    0x2fb41626aa15df56,
                    0x51a3592e98a6c8fe,
                    0x8d58e123abed8f9e,
                    0x083399913ed789c1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf098e34d696ece24,
                    0x37ad7afe1f404415,
                    0x64cb0b1071f247cf,
                    0x187a6b8d3a4e9126,
                ])),
                Felt::new(BigInteger256([
                    0xe57be0df43d42b63,
                    0x4c2dbd6292e89d99,
                    0x92404d04c87e46dc,
                    0x2861735c260175ae,
                ])),
                Felt::new(BigInteger256([
                    0x7bccaee076fdf08c,
                    0x7f801d23ef0f4037,
                    0xa7a940a199e1b03e,
                    0x101aff455ce497d5,
                ])),
                Felt::new(BigInteger256([
                    0x5d62b16d13a2164b,
                    0xaa599cd22041eb8c,
                    0x7671f4c07273ce0a,
                    0x0a6af1081dcd2448,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb88849ecae02b7b6,
                    0xcce2119165fdccd8,
                    0x7ade2f22d3c7f3f1,
                    0x1132099df4d6b571,
                ])),
                Felt::new(BigInteger256([
                    0xb00ddb28aa2c07e4,
                    0x4d5d0d3422b5f635,
                    0x20032fc543740e13,
                    0x16fe7d91e7109bc5,
                ])),
                Felt::new(BigInteger256([
                    0x56beb2067ef17f61,
                    0xe39c02638f514c7a,
                    0x2ed8c0195ebf9c0e,
                    0x25eef6b643b456d3,
                ])),
                Felt::new(BigInteger256([
                    0xa24e2b5defe7f0df,
                    0x43cca63ae41c1b9b,
                    0xa5dd0ca29a2d1a51,
                    0x0ea4438c3f2162f1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8fb4b6aa14999d2d,
                    0x6c1529d63ebb4b5a,
                    0xe56636428cfb5d59,
                    0x2a7e7bcb7029ff87,
                ])),
                Felt::new(BigInteger256([
                    0x256a7c40123e7b5b,
                    0xe21790776e193f43,
                    0x2e4580f02328ec10,
                    0x3e622c2788263d9d,
                ])),
                Felt::new(BigInteger256([
                    0xba0951337ce7349e,
                    0x59b9a282ccbcb17b,
                    0x5b5335b49eefba6e,
                    0x30ea221de0ce60b1,
                ])),
                Felt::new(BigInteger256([
                    0xa9c3f10542bc4008,
                    0x5a19855f781de7ec,
                    0x5818a276e6a8acc0,
                    0x3fae78fc0980bef8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x65c23e8e4e34f41d,
                    0x044a5d464bcf0ab2,
                    0xaf68882c85318168,
                    0x2df1498536a8f5f0,
                ])),
                Felt::new(BigInteger256([
                    0x0c20140fd0e41dba,
                    0x597eef6791c4d8c7,
                    0x7c4a8d7481e6c7d3,
                    0x05b1268b724bb31b,
                ])),
                Felt::new(BigInteger256([
                    0x0c46691e94a6834d,
                    0x9e3770fcab37e766,
                    0x7ff38bc266fa10c5,
                    0x2f9035f01ee833ab,
                ])),
                Felt::new(BigInteger256([
                    0xe72462fae618be33,
                    0x614e66c6710eca64,
                    0x921eddca6ced2afb,
                    0x3fd4858380413007,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x012e8350f22ba422,
                    0x9fe0e6b0b8ca1432,
                    0x194673ead697dbfd,
                    0x0ed8afbf09513a4f,
                ])),
                Felt::new(BigInteger256([
                    0x9c37ea3a9dc41eac,
                    0x944c44e97d8e5eb2,
                    0xfa5d3522ddb87d37,
                    0x18908df545d9befd,
                ])),
                Felt::new(BigInteger256([
                    0x0d284041dd36ed93,
                    0x1241f5a05e97a063,
                    0xa16f655529f629b4,
                    0x0ae0b01a1bbbea74,
                ])),
                Felt::new(BigInteger256([
                    0xed4244f8d0e2c18d,
                    0x47d7a2533890b8b8,
                    0x74d721991042198b,
                    0x2bf029d0a9e7c980,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
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
                Felt::zero(),
                Felt::zero(),
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
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x6a9c5426bac8cb77,
                    0x97d68931091490b8,
                    0xa9391433b87b5f42,
                    0x074b300ed81a3338,
                ])),
                Felt::new(BigInteger256([
                    0xba3385184127f449,
                    0x51768565f3958e3d,
                    0x79e0e52f575f2d0b,
                    0x2628682257a943d3,
                ])),
                Felt::new(BigInteger256([
                    0xd3fe5f791d3fee82,
                    0x7156cb9e4679af23,
                    0x6a5dc37742afc5e3,
                    0x157a378fa166334b,
                ])),
                Felt::new(BigInteger256([
                    0xded987e77614dec5,
                    0x4df5ab31ad0a6d6b,
                    0x4008fe98386281e5,
                    0x2e0acf010781ef2e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa07cb4e2bc93a710,
                    0x4dbe62f6e1e46cc4,
                    0x400c8c285c69aa1c,
                    0x2261ac59f855dd8f,
                ])),
                Felt::new(BigInteger256([
                    0x3c63d685f2b66eb0,
                    0x6a11e140e077d21e,
                    0xd27f09ce968e9969,
                    0x1449d11dffaec97a,
                ])),
                Felt::new(BigInteger256([
                    0x2f34eae26697c906,
                    0x05939589c2413386,
                    0xbcc037e873dc3f43,
                    0x1af1ed62ee441b74,
                ])),
                Felt::new(BigInteger256([
                    0x357ae37277f4dda8,
                    0x56d4d6dca7bb4f9f,
                    0x576a582bdd2eec8d,
                    0x16d4a2340439211d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf6732fde00dedf28,
                    0x0b26219e00f1a9ad,
                    0x1aee1dfd250c3a52,
                    0x042a7d777829c04b,
                ])),
                Felt::new(BigInteger256([
                    0x804dca7eae8663ac,
                    0x851bb54a276e469b,
                    0xa6a9c5b6fcb131ad,
                    0x2bd2f0e73fe15d3c,
                ])),
                Felt::new(BigInteger256([
                    0x888012b76a9f6dc1,
                    0x4f04e73894cbaeac,
                    0x9018cd2173eb269b,
                    0x0c4f151b91a71511,
                ])),
                Felt::new(BigInteger256([
                    0x6811deae940ea425,
                    0x4c6ded826dfebcbd,
                    0xff54c1c0a2575d17,
                    0x237a60401bf7c02a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4ce72f496fd205ef,
                    0x2b29ff3f36b8aa20,
                    0xccc1baf33cc7f9ad,
                    0x2269589118e93399,
                ])),
                Felt::new(BigInteger256([
                    0xda66d5e841589903,
                    0x5315c1bf63cda691,
                    0x2e0e27b05310cc72,
                    0x2a70e6fd04b43f9e,
                ])),
                Felt::new(BigInteger256([
                    0xe73dc352b340471a,
                    0x919f4dfd52997ae8,
                    0x20b8aefe015750e7,
                    0x344123916d88a26f,
                ])),
                Felt::new(BigInteger256([
                    0xd9895e1cfd28981c,
                    0xa8bbc48544883399,
                    0xfeeea0aaa5a44ef3,
                    0x362fd3f504798cdc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x093571f062a988be,
                    0xa17e71501b5a4d7a,
                    0x1cdd4b730eb36888,
                    0x0a670a3e7223757a,
                ])),
                Felt::new(BigInteger256([
                    0x3a2b4dc1be33c970,
                    0x80f725f81a885c29,
                    0x0c9d06b3cb67d27e,
                    0x39b459c3acfcfe7e,
                ])),
                Felt::new(BigInteger256([
                    0xbfcfacdfcd594eb0,
                    0xef4937c5a3f26af4,
                    0x11e098966fcf7ed6,
                    0x2da593341aca3261,
                ])),
                Felt::new(BigInteger256([
                    0x66a005c997650cb9,
                    0xc18b23e8b9c019bf,
                    0xd95686b296078af7,
                    0x13cc15f4a4db2f90,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdbebb49c07003d7c,
                    0x40d10d4819f7fb77,
                    0xfd187d992b324e13,
                    0x09ab75896691f544,
                ])),
                Felt::new(BigInteger256([
                    0x4ea54059c0c55217,
                    0xb61aee55f6194eee,
                    0xebd7a920b5b40397,
                    0x08e9d9a446b38956,
                ])),
                Felt::new(BigInteger256([
                    0x95de555522f5656b,
                    0x80dad5790839e18c,
                    0x9c041c42e210ea0f,
                    0x22539a5334935dc7,
                ])),
                Felt::new(BigInteger256([
                    0xc82589118c01e8a8,
                    0x51faef05751f15ea,
                    0xad83f2a3944abc01,
                    0x3682b3ba229cbf5a,
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
