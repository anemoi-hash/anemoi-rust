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

/// Two elements (96-bytes) is returned as digest.
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

    x.iter_mut()
        .enumerate()
        .for_each(|(i, t)| *t -= y[i].square().double());

    let mut x_alpha_inv = x;
    x_alpha_inv
        .iter_mut()
        .for_each(|t| *t = sbox::exp_inv_alpha(t));

    y.iter_mut()
        .enumerate()
        .for_each(|(i, t)| *t -= x_alpha_inv[i]);

    x.iter_mut()
        .enumerate()
        .for_each(|(i, t)| *t += y[i].square().double() + sbox::DELTA);

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    let xy: [Felt; NUM_COLUMNS + 1] = [state[0], state[1]];

    let tmp = mul_by_generator(&xy[1]);
    state[0] = xy[0] + tmp;
    state[1] = mul_by_generator(&(tmp + xy[0])) + xy[1];
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
                    0x15639f2f82c66565,
                    0x0b96e69953e1980b,
                    0x7782957cab2c3c27,
                    0x1bb04bbe72944a3b,
                    0x32281e946c757694,
                    0x048c01275bf25c4d,
                ])),
                Felt::new(BigInteger384([
                    0xf8718d6da7ac8634,
                    0x3b64f4ebcc064619,
                    0x1a080f2f4585992b,
                    0x8c313ae1d1e199f6,
                    0xfd867f6532eb8dd3,
                    0x14cd05cc66d25290,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x09261f9e3e0776be,
                    0x5bac018165405b61,
                    0x88fe0e96bcd56a29,
                    0x5c0cd31d38248c0a,
                    0xd72cb060ebabfdda,
                    0x18b3dea41beea2a9,
                ])),
                Felt::new(BigInteger384([
                    0x6c954ebb477e8279,
                    0x1fcfa890215171a9,
                    0x435910a178c1909f,
                    0x8ce4ce78bb2b32e0,
                    0x1fa93cb4c931e551,
                    0x096b6ef8130a9dfa,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2e2e8f2b6f87d9f2,
                    0xd59a271c0539bed3,
                    0x3b27e2de9d16ec16,
                    0x6d0f688249d8305b,
                    0x5704bcdb15da00d9,
                    0x17f530d9abd1dc3d,
                ])),
                Felt::new(BigInteger384([
                    0x81895a15d7fe826a,
                    0x34ee6b0d87697420,
                    0x0be223f22f240d7a,
                    0x1fdd2a3de437cf2e,
                    0x30511682ff149dbd,
                    0x167d5ea5875aea1e,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc44af3ddbbb608a8,
                    0x3390856efacae7a9,
                    0x9c78309e574c1bdc,
                    0xa7abfaa8f161f4d6,
                    0xf9f8947047485969,
                    0x086343c0df1b6d38,
                ])),
                Felt::new(BigInteger384([
                    0x97c04c609d451e63,
                    0xfdc71862ae3c169b,
                    0x2e7f41fdc4303684,
                    0x158eae67223d7fd4,
                    0xdbfdb0b2bff3c7cb,
                    0x171e316a8b521ab5,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xdd1e6ec99f72af6a,
                    0x6c732cee7c14a7a1,
                    0x6ad7fa69425afc05,
                    0xf3f7ba5c76cabf0c,
                    0xb4d5240c8e68a216,
                    0x11004e9e105051e8,
                ])),
                Felt::new(BigInteger384([
                    0x3a8e81fde318762b,
                    0xe2151e44ee5939b0,
                    0x5c1f3b41cbbee7ca,
                    0xfdeddca957e63f8a,
                    0xe53a4079d2d9cb27,
                    0x0768eea6734d5f1d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x632bcca2e789cf6c,
                    0x637d0b768e4967a6,
                    0xd6c109b565547e42,
                    0x753ba6dece1629ed,
                    0xf83983cb0d0f58ef,
                    0x08c847ebb427028d,
                ])),
                Felt::new(BigInteger384([
                    0xfd1816abac441856,
                    0x071a8c0018496005,
                    0xf8ca99aa8c3974be,
                    0x105e1692ed3ffc84,
                    0x8ffb5606e5c14e10,
                    0x07fa90ec4161e382,
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
                    0x1efe9ff2c528927f,
                    0x9d5bbede924ac189,
                    0x0d533e2c31061673,
                    0x40af28d076ae518a,
                    0x13eda4ea0204287b,
                    0x0b4a71bbedf263e1,
                ])),
                Felt::new(BigInteger384([
                    0xe528a4074c95e55f,
                    0x4a44ae4f1e8c0a24,
                    0x73ef4d0949e76d0f,
                    0x53c6b895ebac38d8,
                    0x1cf2b95342adfb2f,
                    0x08ec941996e83911,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc1d57c3d8ea41298,
                    0x0c38c341c720e29f,
                    0x37ff01a6b488cbc0,
                    0xb1fccce126d56cf4,
                    0x60c4c91c1906d911,
                    0x009edbedf7fabbce,
                ])),
                Felt::new(BigInteger384([
                    0x4129b55f5209d811,
                    0xcd39927aa2f4a415,
                    0x2674a5ef10a603d7,
                    0xb06cac9384056642,
                    0xe6f87887cecb49ba,
                    0x0dfb12b29ebb5d27,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0be8f49ba7b3c3ab,
                    0xc2b80cc13d2adcb4,
                    0xf6d25c03e8ed291f,
                    0x63c9cc726b963622,
                    0x21782d114796ba06,
                    0x0bc17ba7d2efae78,
                ])),
                Felt::new(BigInteger384([
                    0xd0ded78871ebf5ce,
                    0xa4dd73a7a01aac7b,
                    0x0a26f0fbdc4cdf88,
                    0x54115fa67e51c19a,
                    0x6c6143baf124691d,
                    0x17375b64e3168f75,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x23a50f1f5b8471a2,
                    0x7399b41d0200b7fc,
                    0xe8e171e5d07e2225,
                    0x6864761e90039568,
                    0x400ea592ce7272df,
                    0x03173dadf6db3f4d,
                ])),
                Felt::new(BigInteger384([
                    0x721269e32f15dedf,
                    0x3a5e2c22c04a2bf7,
                    0x964492a0e03977c3,
                    0xbed6fc3afa615a89,
                    0x9d7e543a019f5a10,
                    0x0748a746a458ec15,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x87daa819574d7359,
                    0xac069a52ef8ceb8b,
                    0x72edf84d09b9a374,
                    0xcfbcefa4db9c1ef2,
                    0xcda64c3d8a8a53c8,
                    0x0bc747039bd0832d,
                ])),
                Felt::new(BigInteger384([
                    0xe3b2a55783d967f1,
                    0xef3644347b8d23df,
                    0x423f60082fcfca7c,
                    0x777f3289276965ea,
                    0x775821f455949ba9,
                    0x04d71704e9654da9,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xeeafd2db8fa0b54c,
                    0xaa18a5748097b288,
                    0x0b8a210b3278463a,
                    0x57a168ba2136ab01,
                    0x30dfdf337577f6e3,
                    0x17812843595d784b,
                ])),
                Felt::new(BigInteger384([
                    0xe8dc864f5d5a1ab1,
                    0xfe817e408676ad64,
                    0xcd72e6b267e3acab,
                    0xc28dc6a6451963c8,
                    0xe151080de367b28e,
                    0x019ba55820265a2f,
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
                    0xa2e893d47d9d98ee,
                    0x04ba6e80cee2f26b,
                    0x2ce36ee4b97ddad5,
                    0x9fa5c07027e8f457,
                    0x0b261fa3400861ef,
                    0x10edfabb0cc51802,
                ])),
                Felt::new(BigInteger384([
                    0x9076128eb12ee6c4,
                    0xc52bc09424b36d6d,
                    0x7eedba5067a1ae36,
                    0x66d709d5d3e5fd93,
                    0xd37737d98726e626,
                    0x051468666d17e730,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd401e767b6c6dc02,
                    0xd1c38b4e8c909f4f,
                    0x2d1230eec72b9ff3,
                    0xa1f1ab47ec26cf9b,
                    0x348694c6cfd075e1,
                    0x01157c669ff7c57e,
                ])),
                Felt::new(BigInteger384([
                    0x32fdfa2154106bd0,
                    0xd7ea0ad9a384efd0,
                    0x3719f3121c9a741b,
                    0x2aef4ea0ff86f669,
                    0x80c10eeb2b57a6a1,
                    0x08cb13195b83a83f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x99b1322623bf446f,
                    0xf6e4bbf43e14bc02,
                    0x96b19eb458cb40c6,
                    0x7a5941b08e1f8d14,
                    0x6a5d3fae2f5975f2,
                    0x03f386fa66b5cfd0,
                ])),
                Felt::new(BigInteger384([
                    0x4ba573221efa8a2d,
                    0x02ee9ab46a802ba7,
                    0xcc98c688c3f6573d,
                    0xf9eb66dc20695a3b,
                    0xd2aa5944464da97a,
                    0x00a9073504d0510f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x74fc821876f1ee21,
                    0xbb8d2b1b2d857753,
                    0xb3acbb41f73f8a94,
                    0x2c8a20b63f68c3ae,
                    0xe50555622e49999e,
                    0x1494d591dedb167c,
                ])),
                Felt::new(BigInteger384([
                    0x5e5ae76a80a20d0f,
                    0x89b98d9c19ddff11,
                    0xe33a8928365d25fe,
                    0x99526bc366deaefd,
                    0xc7bbdbc34b625172,
                    0x18fa1ec4bfb802b6,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4a34a70357c58f9c,
                    0x15e558a2bfb0cb4f,
                    0x6d67d452606c4b46,
                    0xc9714ded53864782,
                    0x11293015b7cee372,
                    0x052ad27d205aa7f1,
                ])),
                Felt::new(BigInteger384([
                    0x58693af5f87db3dd,
                    0x51a5b061cf264c0a,
                    0xe5ef6571dc758b5b,
                    0x71370e8a52a29166,
                    0xe61d6e1c6f560a68,
                    0x08196bc4fb257beb,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc6800ccfe317ae73,
                    0x80a212ab49b3cfbc,
                    0x666c6280dc2aba5f,
                    0x3209342a541adc10,
                    0x76c5a38a438fd8f2,
                    0x058152e19ac9daa6,
                ])),
                Felt::new(BigInteger384([
                    0x346c9771e1e2c8ab,
                    0xd95c5fb0e243b3e4,
                    0xa84167849265573b,
                    0xc71435afde98e737,
                    0xffc498e8f175e384,
                    0x090e012e2d10cd8b,
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
                    0x09d5b8f1dffbbbcb,
                    0x7065efaa66f5cd47,
                    0xc38e10e49210411e,
                    0x08dc8896dc2fdcbe,
                    0x66f8e7a00b0a8165,
                    0x0115b99dad74ffc9,
                ])),
                Felt::new(BigInteger384([
                    0xa421847271265e5a,
                    0xa5f79fe8f29f07fb,
                    0x0609dc198bc23073,
                    0x78901b038c45b711,
                    0xa16907199d3be8f0,
                    0x073fdba1c801e6c3,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x39fddbaa5ee7b3a2,
                    0x8197a101d39a7ef0,
                    0x9b4617130060882b,
                    0xf7d04889eb34bc6d,
                    0x3608b29d267fc323,
                    0x12aba29956ff15fd,
                ])),
                Felt::new(BigInteger384([
                    0xecfab17611e02869,
                    0xbc6d4cde9965edb0,
                    0x06754e9726aa8e4e,
                    0xb618942fe26b5c85,
                    0xa1b6cc6f350b8011,
                    0x14214661d001ed9f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x30fc186a61b458c9,
                    0xfcc1f15d13151351,
                    0x2fe32bc5e0b7ef40,
                    0x6e300f68cef2418c,
                    0x0fb1f236bbf4c8e8,
                    0x05459564705671f0,
                ])),
                Felt::new(BigInteger384([
                    0xad9da3f6e2633bbf,
                    0xfc727d6e90aa5249,
                    0x2c5f1e14856635be,
                    0xd64b85adbe4ddd54,
                    0xf20e3db1be373b4b,
                    0x0b3431fde57d34ef,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xbdb450ed7836b2e9,
                    0x91a84655fe997576,
                    0xabc028507697ea49,
                    0x96406133261bfc2b,
                    0xde45bd7c3e76e2d4,
                    0x1286ef46eb4b4eb5,
                ])),
                Felt::new(BigInteger384([
                    0x65c5894571101d8b,
                    0x6fb21a4ab468e9ff,
                    0x6c593487362b0e49,
                    0xfce4971fcc0c81d6,
                    0xee10074f41b8bd6c,
                    0x0a05d97e234ed2ed,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xfb071cef48c0f756,
                    0xb930b9665dfd6363,
                    0x39469f36195761fc,
                    0xabdf6b01f8cb6a50,
                    0xdd640c4e967af843,
                    0x155daa0716a59fc8,
                ])),
                Felt::new(BigInteger384([
                    0x947874d489fff7de,
                    0xa55b232fd9cd12d2,
                    0xf14bd13d18735930,
                    0x647e990950b45347,
                    0x55c9df0359004e18,
                    0x18d3ade8eef0d4e3,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2f593bb3a6dd3fc9,
                    0x335ad20d0e3b3785,
                    0xb6ef318a00f568d7,
                    0xc0319f8a114caa7f,
                    0x764ed55c267b9ffb,
                    0x179d553df4eb75be,
                ])),
                Felt::new(BigInteger384([
                    0x1f210ed92f9df2e7,
                    0x02ba03cd9c1222ef,
                    0x47be2556a6ee3ca2,
                    0x7e88ddba1a2816b8,
                    0x562af434b7d5c9cd,
                    0x044687d5a3e7ebd4,
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
