use super::{mul_by_generator, sbox, BigInteger384, Felt};
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

/// Function state is set to 2 field elements or 96 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (48-bytes) is returned as digest.
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
                *r += *s * mds::MDS[i * STATE_WIDTH + j]
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
                Felt::new(BigInteger384([
                    0x9bd2b150036616e3,
                    0xb76d6fcbb3177f14,
                    0x3993c83d93ea49c1,
                    0x3e809b73f9e764ae,
                    0x205437b9d9f901ec,
                    0x0ed7929b79fc8065,
                ])),
                Felt::new(BigInteger384([
                    0x44b4b57564527293,
                    0x6dd93f26404df096,
                    0x1d2e98ba6f461918,
                    0xad4ead0ab3c462d4,
                    0x85ddd24695619135,
                    0x153445d7cbe1f68b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x52a7073c65678c6f,
                    0x19b9edd7bb180718,
                    0xc27d08019672a93a,
                    0x7f1a5ac27c406851,
                    0x7b0e144d9ed5b9e0,
                    0x155da8b8da39a25d,
                ])),
                Felt::new(BigInteger384([
                    0xbfab6287a32a6057,
                    0x32983d215f92ed0f,
                    0xa53a37c5031d1ca4,
                    0x46debf228b883a47,
                    0xedc9d1d678f81a3e,
                    0x17e63bafcb7dcd78,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x115f9c3b1b3d1e07,
                    0xa458f6079f2de238,
                    0x66b9a2734ef53d20,
                    0x70b3d5a1abc9e631,
                    0xe9bbd0b444406979,
                    0x00005975ec0f4375,
                ])),
                Felt::new(BigInteger384([
                    0x8416adcf7395f8e3,
                    0x1af6b1a0da27760b,
                    0x78cea21cf791e41d,
                    0x8077ae003b290ffd,
                    0xeec2219742525be1,
                    0x0dac616679e778fe,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x180ed4eed77a3555,
                    0xf809736b74549bf7,
                    0xa7e4a88c24618cb7,
                    0x8df708fd71502416,
                    0x6ecd59706c048bdc,
                    0x04f3436e1c92d8eb,
                ])),
                Felt::new(BigInteger384([
                    0x5f69e83a6db2039f,
                    0x6ea94b6031520d09,
                    0x5fa97a9a593e18fa,
                    0x423737f3b94f4e08,
                    0x4d89272758007e02,
                    0x023d54c67f09c362,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3a1e48f763a14756,
                    0xca3931efea2fb075,
                    0x2a630db5cdf28023,
                    0xd662c70c1a87e179,
                    0xfc68959a540463fb,
                    0x1586e680e0240ceb,
                ])),
                Felt::new(BigInteger384([
                    0x45e83cb9a185d14a,
                    0x44efe81349bc9d9a,
                    0xf01c80e4a7b9b807,
                    0x5c23d9ea6295f8a7,
                    0x310d73eea46e939d,
                    0x1718542395669097,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x78de57d92bb3656c,
                    0xa408e40c819b9c6c,
                    0xbdc5feb686c09c09,
                    0x2cda32afd3a90599,
                    0x80739efe7216c37c,
                    0x0049efc39ce60bd0,
                ])),
                Felt::new(BigInteger384([
                    0xb30f01034af21062,
                    0xdb5a7c4c1b748300,
                    0x9f2c436abea67f80,
                    0x620d7f3289a3d86a,
                    0xa7b1d09b7393db12,
                    0x0c13573b7fcd7e3c,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger384([
                    0x1804000000015554,
                    0x855000053ab00001,
                    0x633cb57c253c276f,
                    0x6e22d1ec31ebb502,
                    0xd3916126f2d14ca2,
                    0x17fbb8571a006596,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger384([
                    0xf64900000018553d,
                    0x40f4005f6f0c0013,
                    0x9313f019a789cfb3,
                    0x59fb77168f0da76b,
                    0x951d2d06cf6bb694,
                    0x15b1e4359a873e00,
                ])),
                Felt::new(BigInteger384([
                    0x321300000006554f,
                    0xb93c0018d6c40005,
                    0x57605e0db0ddbb51,
                    0x8b256521ed1f9bcb,
                    0x6cf28d7901622c03,
                    0x11ebab9dbb81e28c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x221a96dee879506e,
                    0xf474857d61fd8d8e,
                    0x0cb62c7142732926,
                    0xdab2c2b279e8e05a,
                    0x66bfd291dacd2598,
                    0x14a7b8b22c4d8158,
                ])),
                Felt::new(BigInteger384([
                    0x8501827008bb463c,
                    0xe000d5f0f9385e8d,
                    0x651c1aea973d56a5,
                    0xc16f501811680aeb,
                    0xbd0d7da3633e27ed,
                    0x100d54ca211c6624,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4c220000000b554a,
                    0xed28002c72d80009,
                    0x4b84069f3c7f4f33,
                    0xa827f857a8538294,
                    0x0653b9cb0ff30b64,
                    0x0bdb9ee45d035f82,
                ])),
                Felt::new(BigInteger384([
                    0x43f5fffffffcaaae,
                    0x32b7fff2ed47fffd,
                    0x07e83a49a2e99d69,
                    0xeca8f3318332bb7a,
                    0xef148d1ea0f4c069,
                    0x040ab3263eff0206,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe0bdc1c5e2b55e06,
                    0xb76f0676228e5e74,
                    0x2fbaf59f66d056ea,
                    0xc8ff9cd89ca99aab,
                    0x6dbf68621e347688,
                    0x0fed36e07d855e85,
                ])),
                Felt::new(BigInteger384([
                    0x5ae7300061178397,
                    0x8428e8ffdc7a4d49,
                    0xa5bf09c57317f5ea,
                    0x8dd74862b2f69511,
                    0xe4d2dbc19d0eb2b2,
                    0x1008b4eaf6509d29,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x160f76e4efc25df3,
                    0x175d26ebfec0c569,
                    0x408ba2a98dd3b39d,
                    0x9d64978ade27493b,
                    0x68800917669696ca,
                    0x1282988896d35887,
                ])),
                Felt::new(BigInteger384([
                    0x9132f43c7c83b5f5,
                    0xfed8a739bfc5bef5,
                    0x043f4681f3c2d76e,
                    0x46faa5896847cd80,
                    0x9bd8fafba64bfff8,
                    0x15c744284d70fcf7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x00c28cc370ccfcf2,
                    0xbf600b6790d8a416,
                    0x8968a2a8eafec92f,
                    0xbfeb206d0b865c48,
                    0x65ca908dbe272ff8,
                    0x16f17ebc0b0e855d,
                ])),
                Felt::new(BigInteger384([
                    0x96627d39a1935a2a,
                    0xb7e6a3f8984bf689,
                    0xed47ba799c513e4f,
                    0xce0cf3518393b2f4,
                    0xc0a12a9de5b516fd,
                    0x02c7e79dcfb708ae,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x43d4099570773d56,
                    0x52e92fc67a472e1a,
                    0xaf0c4aeacf877dde,
                    0x5340b9ec5bcd20e7,
                    0x2bb0f3281ce398dc,
                    0x02b70cee9bc3cc41,
                ])),
                Felt::new(BigInteger384([
                    0xb2a267fddfd8f2d6,
                    0xa7999e6f67f9d13f,
                    0x0b88339bec563806,
                    0xef341415970f3e8f,
                    0xcf2d5c8d51952936,
                    0x184dd74a3fe8ce0b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x095181e5fa3564c7,
                    0x9efceb44ab69c43f,
                    0xa2154426dda44dfb,
                    0xd07e1e4f951bad64,
                    0xbece3bb9dc259c07,
                    0x0bc79fa1f7322d67,
                ])),
                Felt::new(BigInteger384([
                    0xac8436e24922461f,
                    0x924ca1f2968934d7,
                    0x0ec854e3c50451e0,
                    0x28f81694a4417f15,
                    0x04be49314253ba3c,
                    0x166433f5ce9b0657,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3751fca2bd3a713a,
                    0x8d7cb0cfc201afd1,
                    0xbc3ce9f65ad964e0,
                    0x2b90dddb359a45f7,
                    0x3d73e02585cf9fb3,
                    0x0c4b0359c599c669,
                ])),
                Felt::new(BigInteger384([
                    0x659d353f457b839c,
                    0x4795b198763e1729,
                    0xc8d795a0b7283136,
                    0x1d4980066ae6da07,
                    0x94fec801f7f16d39,
                    0x02bf12d9430c8d1c,
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
                Felt::new(BigInteger384([
                    0xb9c33b9b449a5c00,
                    0x18e525a50211bd7e,
                    0xa9060e9c527798e4,
                    0x4f85fe6fae4173e0,
                    0x93fc942f746a8505,
                    0x12825e202546581f,
                ])),
                Felt::new(BigInteger384([
                    0x990f52f822452551,
                    0xd97a991567030f6e,
                    0x6849016c83bc9bf1,
                    0x1c37cae87003f35e,
                    0x0a7bd9ac9f093d4b,
                    0x021137c76cbe4580,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x06289b1781df1dc4,
                    0xc49082fd2638cba2,
                    0xd96a7f8eff77f096,
                    0xac883aa86e83b3e1,
                    0xb07e53761c3af095,
                    0x01e8c2b54c1a08cb,
                ])),
                Felt::new(BigInteger384([
                    0x1d275ad9088cdf98,
                    0x900dffa789117315,
                    0xd71f7a946fdd8387,
                    0xeba8fbbef8a5b112,
                    0xf16b4c357747650d,
                    0x1068e4faa76ccc0e,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x941c20e9dd5290ac,
                    0x4ab21d94f65fda02,
                    0x140367ae15a93d84,
                    0x66009cbd47baa067,
                    0x17cdbfff0a6be4af,
                    0x0d856cf8a3312db8,
                ])),
                Felt::new(BigInteger384([
                    0x76d7198ef89eeccd,
                    0xee7fcd7d9fecede1,
                    0x0284d39dff0fe8c4,
                    0xf264981a9fcf9b82,
                    0x2d80ddc9fbb65b00,
                    0x15a82c45ac49e19f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x44ae60b46e84f1ca,
                    0xc77f520006cb4090,
                    0x55500184706f8217,
                    0x2bd52d180ee576af,
                    0x9b8c181f56ee744d,
                    0x067e7c25c4cdd795,
                ])),
                Felt::new(BigInteger384([
                    0x19a977957bf810ca,
                    0xfcb7ad67110c0f04,
                    0x1a56a024aaf56155,
                    0xa965e95be877fad9,
                    0x5f9914903c71d453,
                    0x06245ddc12b3799a,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4c9ba1c0036b54ac,
                    0xeed949c2028167ad,
                    0x996b8e031ce9ef5c,
                    0x75023d286bc69564,
                    0x4a2af5452ae0ed0d,
                    0x0501db16e972856a,
                ])),
                Felt::new(BigInteger384([
                    0xecdfcfb458fe1541,
                    0x2f683cdf14350b77,
                    0x270866c9e59b3eb9,
                    0x182e6be28aa71e11,
                    0xe04cbdaf21bda410,
                    0x0bfdc7494d36d775,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x81d06c1adfeb6fa6,
                    0x65dc02c294102d28,
                    0x4ef9784ed22ca575,
                    0x6f1747214d086c27,
                    0xcc5dce0499083df3,
                    0x0c5d48d1c2a0fded,
                ])),
                Felt::new(BigInteger384([
                    0x210bb1be07f76737,
                    0x77a355493816976e,
                    0x47214bc687aaf11b,
                    0xe5fe0d0cb843e720,
                    0xb7f9c5d14a88d7b8,
                    0x0183a450f57cc94b,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger384([
                    0xee1d00000009aaa1,
                    0x86840025e97c0007,
                    0x4f7823c40df41de8,
                    0x9e7c71f069ece051,
                    0x7dde005a606d6b99,
                    0x0de0f8777c82e085,
                ])),
                Felt::new(BigInteger384([
                    0x984400000016aa94,
                    0xda500058e5b00012,
                    0x97080d3e78fe9e67,
                    0x504ff0af50a70528,
                    0x0ca773961fe616c9,
                    0x17b73dc8ba06bf04,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x321300000006554f,
                    0xb93c0018d6c40005,
                    0x57605e0db0ddbb51,
                    0x8b256521ed1f9bcb,
                    0x6cf28d7901622c03,
                    0x11ebab9dbb81e28c,
                ])),
                Felt::new(BigInteger384([
                    0x6631000000105545,
                    0x211400400eec000d,
                    0x3fa7af30c820e316,
                    0xc52a8b8d6387695d,
                    0x9fb4e61d1e83eac5,
                    0x05cb922afe84dc77,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger384([
                    0x321300000006554f,
                    0xb93c0018d6c40005,
                    0x57605e0db0ddbb51,
                    0x8b256521ed1f9bcb,
                    0x6cf28d7901622c03,
                    0x11ebab9dbb81e28c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xebe1e18b8924a6a2,
                    0xcbda57cfd017dc5b,
                    0x7998117559f0d0c7,
                    0x87f594408e495a9d,
                    0xa8f44788b27cff9b,
                    0x16a4cdaefec2e31f,
                ])),
                Felt::new(BigInteger384([
                    0xb6d4160f348ec7ea,
                    0x528348b655dec826,
                    0xf44851b640ed475d,
                    0xc7aba7e4991195d9,
                    0x1148c107c0b78faa,
                    0x1559c13b30c42525,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x867850c992f93249,
                    0xc600824d8707b1cc,
                    0x2078a216e8820181,
                    0x1f62e6a16c4a0348,
                    0x4839442ac77e0dda,
                    0x08b97ac06173ba4f,
                ])),
                Felt::new(BigInteger384([
                    0x7018fc6c2e7f997f,
                    0xfd630443e5ccd6ae,
                    0xb0dfec214a309066,
                    0xc5f77d7cddb4a4e3,
                    0x36c22cd4c2f7d3ea,
                    0x07dac89130d45a13,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0dcc5407ce9114f0,
                    0xea59b892d391b5c6,
                    0x4aab69a8266722c5,
                    0x81db35e8a04fb1ec,
                    0xdc982c267b414102,
                    0x04d3a1af88c523c1,
                ])),
                Felt::new(BigInteger384([
                    0xd870c19e95c16c02,
                    0xa4873ea495bc596d,
                    0x30aad44d552d382c,
                    0x91a3b866ece9ec9b,
                    0x9b958e60aeed302e,
                    0x054e5dba84544288,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x78014fdf6675135e,
                    0xc0eeacce28e35e98,
                    0x89fd41cdc65a44c3,
                    0x7ea0ffcfdfd56c61,
                    0x5abe413fcfd21cf4,
                    0x12c737ddea34caca,
                ])),
                Felt::new(BigInteger384([
                    0x4fad175448e28cdb,
                    0x5fe90704b17ecc35,
                    0xc720511f40f8f4b9,
                    0x42309d76b49dc0dc,
                    0xc9f9ef5998ca6165,
                    0x11b1bbadad9d2894,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6c5c4128b567d483,
                    0x2efdc38179977e9d,
                    0x804b88f5f16f76ab,
                    0x40e7c9688d8fbec7,
                    0xbfa8c8ed2b108856,
                    0x02fc57bf4a604dbb,
                ])),
                Felt::new(BigInteger384([
                    0xc5985205c3cdbe47,
                    0x8d63c3e2076408b2,
                    0x279f78b5c87a2c0f,
                    0x99fdfeb3a5c69ba0,
                    0x5f9e4f8977deb4bc,
                    0x11f676c7e1f772ed,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc3e7cf96efda3e14,
                    0x5522ad55043d5c04,
                    0xdd3c0fdbe18287ac,
                    0x3b13613abd903a67,
                    0x3c5159a72e19ed65,
                    0x0f649173ad9a9085,
                ])),
                Felt::new(BigInteger384([
                    0xeedc50ebe7ac38b4,
                    0x033caff48f3d4f77,
                    0x9a6898dd53ff0a50,
                    0xf7ad83fd3fdf4930,
                    0xe580d169637105ab,
                    0x064bb54e173203bb,
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
