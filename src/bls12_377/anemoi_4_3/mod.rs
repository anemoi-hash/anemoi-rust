//! Implementation of the Anemoi permutation

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

/// Function state is set to 4 field elements or 192 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 4;
/// 3 elements of the state are reserved for rate.
pub const RATE_WIDTH: usize = 3;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 2;

/// One element (48-bytes) is returned as digest.
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
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger384([
                    0x366f547a173ce55f,
                    0xf5be288552382c5e,
                    0x41a84e2f7c1f0590,
                    0xcf1b80bba093b75f,
                    0x7d71995160b97683,
                    0x0028f2cc3d955547,
                ])),
                Felt::new(BigInteger384([
                    0x45df808c193ee1f8,
                    0xd14cc057ba32a9f8,
                    0x9e052572d1ea125f,
                    0xf2634854f0a0aea9,
                    0x051adb0db8a3583f,
                    0x0040e1286de7f4b0,
                ])),
                Felt::new(BigInteger384([
                    0x2976fb86da3f1fd1,
                    0xac16fecf08c55b32,
                    0x3f9920176b66d9f2,
                    0x62ffb54f38296849,
                    0xd57c80424a494c54,
                    0x018531a07f3d9ad7,
                ])),
                Felt::new(BigInteger384([
                    0xc703e32f5fd9119c,
                    0x584c867c78a98a58,
                    0x7deb8df1aa346d91,
                    0xd04edc7ecd18837a,
                    0xea93810a7cc49b83,
                    0x01a6451300cfd7fc,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x83a29cc2c4f87679,
                    0xcbdc178df81d0370,
                    0xcba68580b2687dd3,
                    0x48a96247c36d98a9,
                    0x706002802089a2a5,
                    0x0067a03dc5ed45fa,
                ])),
                Felt::new(BigInteger384([
                    0x5aa40db3b0b988b3,
                    0x615ce2869eef4084,
                    0xd45c9fb7ac23e502,
                    0xa1ee15d5b53baf2b,
                    0x702afbb7b717d2b0,
                    0x00689bf8764843cc,
                ])),
                Felt::new(BigInteger384([
                    0xcecaa8bcbbc6614c,
                    0x9381f9abfdfaf930,
                    0x02d869cfdac79b0e,
                    0x0c1f733c2c3d612e,
                    0x2132393af3aae278,
                    0x00d6fa03b8e1c902,
                ])),
                Felt::new(BigInteger384([
                    0x121af983594ae7f0,
                    0x2ff6df050e12244c,
                    0xf4be6abd5f51148d,
                    0x194f0ec99633a0e9,
                    0x38fbe1baa98a9395,
                    0x00279d44e6f7aea8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xa8412a7cb9f4a735,
                    0x5529c97ffd0af6ee,
                    0xc14abc346e6de861,
                    0xd04734aa20cc342f,
                    0x33d662e82b47b651,
                    0x0187a5997327cf14,
                ])),
                Felt::new(BigInteger384([
                    0xcbb20a1192d011f2,
                    0xbf114cf584101294,
                    0xf7676848b97b5520,
                    0xbb6fc4d4f220cc83,
                    0x3ad7fcbbcbff1687,
                    0x00b6f4aecb6cd473,
                ])),
                Felt::new(BigInteger384([
                    0x5c933139bc0891c6,
                    0x40b9dc6c740ea9c6,
                    0x4ad5b0c7ae92b944,
                    0x53625794ea680fa5,
                    0x0b1257f82437abfb,
                    0x0005981be399438d,
                ])),
                Felt::new(BigInteger384([
                    0x4c123af4cfadfeac,
                    0x7cef29b55b7792a7,
                    0xa9c08e5615593512,
                    0xd5b2edb491707ca3,
                    0xdd51b33cf283f79d,
                    0x00905d1a3c33b133,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x199b1080a894f48a,
                    0x0093d35f337379eb,
                    0xff91276717435cb8,
                    0x25e64e5e922052f5,
                    0x6a5c641868313c14,
                    0x019f4c8fbdb94a79,
                ])),
                Felt::new(BigInteger384([
                    0x08937daa7b4f6b00,
                    0xd8b2661e59cdb546,
                    0x0e7c52e6b1288d3c,
                    0x1a924e301b95ca63,
                    0x4f8eaf606c3dbeb2,
                    0x005d51fc0de64e13,
                ])),
                Felt::new(BigInteger384([
                    0x1e5475843dfce982,
                    0xf47e8d2e8294e10f,
                    0x957fe1113208bd79,
                    0xc344bcfeb6e760fd,
                    0x70fb8670cb3d926a,
                    0x01969f90a4615869,
                ])),
                Felt::new(BigInteger384([
                    0xfdff813058b591cc,
                    0x86f2a8e15a884e44,
                    0x4795d8cfd8d14c1d,
                    0x90603374a87373f7,
                    0x7214bd8676ace6c4,
                    0x0072b10988344468,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4a1da1b265f2f51a,
                    0x1acd8b4c7130ddf5,
                    0xae73f9c9d91021ad,
                    0xfeff618be7a7f0c7,
                    0xd416ab9c637e8213,
                    0x00732b1304ee9b0f,
                ])),
                Felt::new(BigInteger384([
                    0x768fe681cbfe428e,
                    0x8bde93ae104971a6,
                    0x1064ed5c14cb970a,
                    0x844b16bbc27d2516,
                    0x4da8befddde87b9e,
                    0x00c8d876eebc1f2f,
                ])),
                Felt::new(BigInteger384([
                    0xe4e9baac921e8f24,
                    0x2cd38340fe2f45b1,
                    0xb15d545ce79053ed,
                    0xa528d34ffde4a75c,
                    0x5ea33152da8ba505,
                    0x01abd178c847a156,
                ])),
                Felt::new(BigInteger384([
                    0xe1e6b61b374618a3,
                    0xe1bd2e62ba03aacc,
                    0x3963bde3c5227566,
                    0x2177da684fcbf217,
                    0x3de2331e9ac7b1b1,
                    0x013f8174d32490e9,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x25a22a322aebf4eb,
                    0x21a6a670d877143c,
                    0xa418e23e1d77f7e0,
                    0xd81fecb019d3dd69,
                    0xf3d79c5f193e2993,
                    0x006911f1319412bd,
                ])),
                Felt::new(BigInteger384([
                    0xa7abdf9d609db3f2,
                    0xaeb1efdba0e198ad,
                    0x7131b7d0d66c7b0d,
                    0x1445075f0a71e521,
                    0x4067b892ec866dad,
                    0x00435bbb0f55239a,
                ])),
                Felt::new(BigInteger384([
                    0x331c8327147ea9c6,
                    0x9c556dea4b20d097,
                    0x149ad9f8e7a207db,
                    0x09c23a3390ef2274,
                    0x31a32aaaccf113a7,
                    0x00ee073cce92d7af,
                ])),
                Felt::new(BigInteger384([
                    0x70f585059712617b,
                    0xbd74d5b958cdfe03,
                    0xb17b874643129b14,
                    0x60337a93814de354,
                    0x8aad0ecd200497ab,
                    0x01055ef2220cab7c,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger384([
                    0x56dcddddddddddd4,
                    0x2db2015f37777772,
                    0x8a5a595c4be8b110,
                    0x2041bbb36e056126,
                    0x7e422da67ad9b5fd,
                    0x007c276e8cf025e2,
                ])),
                Felt::new(BigInteger384([
                    0x56dcddddddddddd4,
                    0x2db2015f37777772,
                    0x8a5a595c4be8b110,
                    0x2041bbb36e056126,
                    0x7e422da67ad9b5fd,
                    0x007c276e8cf025e2,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger384([
                    0x59b3fa50b6287770,
                    0x8d8cc2314dc632d6,
                    0x29b1a14daf4e27f8,
                    0xedb8547b775eb818,
                    0xd8e661a68b195234,
                    0x0105b860faad5b24,
                ])),
                Felt::new(BigInteger384([
                    0x59b3fa50b6287770,
                    0x8d8cc2314dc632d6,
                    0x29b1a14daf4e27f8,
                    0xedb8547b775eb818,
                    0xd8e661a68b195234,
                    0x0105b860faad5b24,
                ])),
                Felt::new(BigInteger384([
                    0xa578190a443f6866,
                    0xba8dc4fbbf9b29b6,
                    0x7eb0b8ad3adf4bee,
                    0x878939a768b75022,
                    0x5a0c806ddc201082,
                    0x00b44682633de315,
                ])),
                Felt::new(BigInteger384([
                    0xa578190a443f6866,
                    0xba8dc4fbbf9b29b6,
                    0x7eb0b8ad3adf4bee,
                    0x878939a768b75022,
                    0x5a0c806ddc201082,
                    0x00b44682633de315,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3f83be968f6712f4,
                    0x7a14ed2c157a3e9f,
                    0x13379f76f22125ab,
                    0xd2b0fd505a8223b8,
                    0x90580d58fcb34c3f,
                    0x0073a0b34fa4fca9,
                ])),
                Felt::new(BigInteger384([
                    0x3f83be968f6712f4,
                    0x7a14ed2c157a3e9f,
                    0x13379f76f22125ab,
                    0xd2b0fd505a8223b8,
                    0x90580d58fcb34c3f,
                    0x0073a0b34fa4fca9,
                ])),
                Felt::new(BigInteger384([
                    0x6af47e2abade4fc8,
                    0x1a826133f52d710b,
                    0x6c3926ec49ea6ba8,
                    0xbfd567223db99dfb,
                    0xca056d88e9e47a82,
                    0x0036eb7a315a0db0,
                ])),
                Felt::new(BigInteger384([
                    0x6af47e2abade4fc8,
                    0x1a826133f52d710b,
                    0x6c3926ec49ea6ba8,
                    0xbfd567223db99dfb,
                    0xca056d88e9e47a82,
                    0x0036eb7a315a0db0,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xea911dddddddd44f,
                    0xce8327424777727f,
                    0xe774a906518e4834,
                    0x527cf56b51022fb4,
                    0x6e646cdc5f7b965d,
                    0x00eb6a2e45f61af1,
                ])),
                Felt::new(BigInteger384([
                    0xea911dddddddd44f,
                    0xce8327424777727f,
                    0xe774a906518e4834,
                    0x527cf56b51022fb4,
                    0x6e646cdc5f7b965d,
                    0x00eb6a2e45f61af1,
                ])),
                Felt::new(BigInteger384([
                    0x823ac00000000099,
                    0xc5cabdc0b000004f,
                    0x7f75ae862f8c080d,
                    0x9ed4423b9278b089,
                    0x79467000ec64c452,
                    0x0120d3e434c71c50,
                ])),
                Felt::new(BigInteger384([
                    0x823ac00000000099,
                    0xc5cabdc0b000004f,
                    0x7f75ae862f8c080d,
                    0x9ed4423b9278b089,
                    0x79467000ec64c452,
                    0x0120d3e434c71c50,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x06c2480cc86ac4e3,
                    0x6c156ed18c5e25e5,
                    0xd208c0e0320ce2c1,
                    0x103cfe9d510ee3a7,
                    0x082aa1069912d9cc,
                    0x00acca1ceca91b18,
                ])),
                Felt::new(BigInteger384([
                    0x43e379fda4606059,
                    0xae5425775e3b387e,
                    0xdf81566b6836a5e0,
                    0xd1702616227ef2ad,
                    0x2c05af53f84e9d27,
                    0x001c6137e4c10368,
                ])),
                Felt::new(BigInteger384([
                    0x632edd43c85cdf66,
                    0xd711a265ce637ffc,
                    0x808fc3470eed5697,
                    0x46fed0bc65ccdd76,
                    0x12c88aa530282b07,
                    0x00c8c9c5bff0f40a,
                ])),
                Felt::new(BigInteger384([
                    0x6b0938f2df9d7843,
                    0xb2fb4f8204a23837,
                    0x3f797c3b21cf31f7,
                    0xeb3b6c33497d4484,
                    0x1f7d458dad678597,
                    0x007aa98b070df89e,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb504b8beef9a8e66,
                    0xe6599f82dd19533f,
                    0xeb8bd9395e58fbec,
                    0xfafa3b8f24c798dd,
                    0x24947a9ab21e69cb,
                    0x00b97e8df2085b60,
                ])),
                Felt::new(BigInteger384([
                    0x1b27c59f0cad83f9,
                    0x5b0a94ce743c0ef5,
                    0x92b303fd2c3189f3,
                    0xf28a62a9fb7dbe97,
                    0x114000007393a68a,
                    0x00baad66b9b220b4,
                ])),
                Felt::new(BigInteger384([
                    0x5b8cc5762c93241c,
                    0xbab5d7e1bc0d8e72,
                    0x8255576e8c1bcbbe,
                    0x9610938e240f50c7,
                    0x8840c3178a6e8d2c,
                    0x00bd28df9c22fd6a,
                ])),
                Felt::new(BigInteger384([
                    0x89def281a733c4fa,
                    0x098d4ac5b5e20b0d,
                    0xe6b46fdc72c950fc,
                    0xe918eddd90e0c046,
                    0x8d3544850a21b328,
                    0x0153083d77735282,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x533216491f2d85fe,
                    0xa052ca5247e853f6,
                    0x1e06b651070d6982,
                    0x3858120dde532941,
                    0xbc28e678ece75c09,
                    0x013c1c8d09047a68,
                ])),
                Felt::new(BigInteger384([
                    0x4d13bbd9b9f0842c,
                    0x7cb6050c9a5d5031,
                    0xdce2452f69f2f9e2,
                    0x8f97fca4f2065f21,
                    0x9094b6845a9b9bff,
                    0x0047c63944a70d11,
                ])),
                Felt::new(BigInteger384([
                    0x01fe3d396415fc85,
                    0xe48ec493bb2ae32c,
                    0x7b56b28bbeba591b,
                    0xdd15e5bae4c82e28,
                    0x9740b5882fd92d19,
                    0x01011e2316e14c6f,
                ])),
                Felt::new(BigInteger384([
                    0xc3a30c0df69fe538,
                    0x68c287c593c49c8d,
                    0x3ab793a117475266,
                    0xa82532b7c9b5c874,
                    0x14da88a3174ef369,
                    0x0016dac3afe39a55,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xa72131b0481d6fc1,
                    0xe158c93c03a7bce6,
                    0xa206e2f6db7046da,
                    0x8a6a6a5ce58130b5,
                    0xe498ec97b42fc1b2,
                    0x0138352deb223d67,
                ])),
                Felt::new(BigInteger384([
                    0xf691820c3e02eef7,
                    0x8867b80aad40e87c,
                    0xc1f9bb78a470abb4,
                    0xe6398acc949eecf3,
                    0x2a4e5da2f4f113b3,
                    0x0199a85ac317dc5c,
                ])),
                Felt::new(BigInteger384([
                    0xb3867f84b61fd637,
                    0xf641cd10cbd31e12,
                    0x49034b69c3fe8024,
                    0x0129cc3e78a3c527,
                    0x5ef98f9fb89efe89,
                    0x011f862cff3ae912,
                ])),
                Felt::new(BigInteger384([
                    0x7b983db211a583ac,
                    0xa569e307ff105d5b,
                    0xeda0697068fe6e61,
                    0xb1f2b45c86910399,
                    0xa7fdf0996c4cd688,
                    0x0080c3a3b500110c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xf3ee05910fbf7696,
                    0xc9e7bf5b1dce02b1,
                    0x4cee7bf9e9300cab,
                    0x161dff2463645547,
                    0x95183d1366a96b5a,
                    0x009079699c25f0db,
                ])),
                Felt::new(BigInteger384([
                    0x4d78e1b079009aa8,
                    0x48c0582de1faa238,
                    0x386aefbd3cb0eead,
                    0x92687f29c581275d,
                    0x444ac238d0449ad9,
                    0x018336c793e0594e,
                ])),
                Felt::new(BigInteger384([
                    0x446be52c607f8828,
                    0x39bff3773c0bf8f4,
                    0x8f941fbc32fd46e1,
                    0x24a0fa5c877910e3,
                    0xbe19fa158997e49a,
                    0x00eb3741cd213e0e,
                ])),
                Felt::new(BigInteger384([
                    0x75b0293cd925eb8c,
                    0x5c1c66235aa01aa6,
                    0x3d42500a4250dbfd,
                    0xd7e2953415cd0cae,
                    0x50b84d4fa6932c27,
                    0x002e5f169b042c4d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x548274b95f416927,
                    0x42f4d8df07f15980,
                    0xffc3025c49e54c37,
                    0xa87f87354440047c,
                    0xb9b5131cd144464f,
                    0x01061991a2536119,
                ])),
                Felt::new(BigInteger384([
                    0x1d5bbc6767ec3730,
                    0x962038cb39cbbd54,
                    0xe287e9ce016fbb01,
                    0xa79b7caa525f050c,
                    0xe72bb4e627e52e28,
                    0x001d529ddcc4499e,
                ])),
                Felt::new(BigInteger384([
                    0xbb12e163ba24613a,
                    0x0e1f24f763ca81f3,
                    0xa49a1d68adb7c33c,
                    0x366a7856391c41b0,
                    0x53e4dad4faead80b,
                    0x005bd313bc168bac,
                ])),
                Felt::new(BigInteger384([
                    0xa359fa12fce89d92,
                    0x50aff4640d743b65,
                    0x082e98ea78366d9a,
                    0x97090618aa01f815,
                    0x7151d3921180bf62,
                    0x01039e42b184f6e2,
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
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger384([
                    0xbecfd275216b499c,
                    0xff263e83132796dd,
                    0xb6f0176a5c19b303,
                    0xf6bcf7375d5ea974,
                    0x58936a31d767a85c,
                    0x003e0f0258533084,
                ])),
                Felt::new(BigInteger384([
                    0x80eca2504bfb1657,
                    0xb6f0f030c068463e,
                    0x60948ef79e5df1ca,
                    0x26e512100b254e01,
                    0xf543f0cac48dcccd,
                    0x0073efae9955490e,
                ])),
                Felt::new(BigInteger384([
                    0x6c72832efec3453a,
                    0x2b1ed2ffeabfcad3,
                    0x6c8cfff9e380e32d,
                    0xf2890f0d64ca4cbe,
                    0x03dcfebb64f547dc,
                    0x006b7f059822dd25,
                ])),
                Felt::new(BigInteger384([
                    0x04641ce8a78b0246,
                    0xd9e9fbd96b902b64,
                    0xd32179d3733c9903,
                    0x13a79a3a5bc31d9a,
                    0x61e2e900c8133481,
                    0x015b3abe63a5ec32,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x1acb998cc9d2d723,
                    0xd537f183af7b6abf,
                    0x91ecbfeaee0d39be,
                    0x25ae351f361d09c0,
                    0xc491ac44fe69e073,
                    0x019275f80df68eb4,
                ])),
                Felt::new(BigInteger384([
                    0x8b5191c8bca57367,
                    0x772345404bf488f5,
                    0x8513c2ef81f99962,
                    0x0be9319c6e729151,
                    0x60539a22ea8c1619,
                    0x004eff20b7fa84b6,
                ])),
                Felt::new(BigInteger384([
                    0x001a15568ceae294,
                    0xdb616822702c1200,
                    0x551c78e59d55a54e,
                    0x09559982c0117945,
                    0xb15b3b72b1fcbbdc,
                    0x00bf477cb4a20f7e,
                ])),
                Felt::new(BigInteger384([
                    0x3f7bebef353ffe8a,
                    0x26f24f036c30a660,
                    0x36394861bc985809,
                    0x55ca0d43d2f9248d,
                    0xde89bd61b81407ec,
                    0x009bfd261bdf762c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x02e685749866a805,
                    0xcf52672e0148bb34,
                    0x77f52fbc6375d4fb,
                    0x8d8bd74988ff8e3f,
                    0x5d0f3128fe172198,
                    0x00e698fbc8a748cb,
                ])),
                Felt::new(BigInteger384([
                    0xe764105fddd360cf,
                    0xa5704551f3bc6c02,
                    0xc5a90393ebe0302a,
                    0x9b315150e7c3d3c0,
                    0xda5610bbb149a177,
                    0x00fbe56d441d2e9b,
                ])),
                Felt::new(BigInteger384([
                    0xbc76c52a277b571d,
                    0x6b4bff916ccd3eb4,
                    0x5faea8da10a62078,
                    0xdb7bc590724f6687,
                    0x838a4ab5f87f6f95,
                    0x000554e5d52a3469,
                ])),
                Felt::new(BigInteger384([
                    0x930c64a34312bd85,
                    0xc2b67bf5a2d1fef1,
                    0x0cac50879c758d3c,
                    0x3408ff31837946b1,
                    0x72a8d44b693a91a3,
                    0x00925523e23607c9,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x673e7e76ee762603,
                    0xc5fd77a903e48102,
                    0x47da8db34b45fa93,
                    0x6052291fa9d2622c,
                    0xd027534232517131,
                    0x00864bae487719eb,
                ])),
                Felt::new(BigInteger384([
                    0xe4fc9e3626f8ff02,
                    0xca4a20caa6ce1610,
                    0xb1bbfb8621c64bde,
                    0x03413f03bc965a27,
                    0xacd645a0b0d0beb3,
                    0x018c2c83245b935e,
                ])),
                Felt::new(BigInteger384([
                    0x51a6e641b9f42ef3,
                    0x9fd1241391310db0,
                    0xe9cd1950febaff16,
                    0x17bbaa13431f0614,
                    0xc2c6931581977cd4,
                    0x00e8a978bf2f2e8d,
                ])),
                Felt::new(BigInteger384([
                    0xa80f3fe11620ec70,
                    0x2d135f9ad8326b07,
                    0x19cc4294ce25080e,
                    0xfa21c1487e819f67,
                    0xb2979d4c064cb870,
                    0x005a70d1a3a402c7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd2fcc0d6bbbfc8ca,
                    0xf04eef7a25067d8c,
                    0x161c24b346a6f067,
                    0xc9eb435876c3cd89,
                    0x729bb181f0e4ec87,
                    0x01a37c6ff6ae618e,
                ])),
                Felt::new(BigInteger384([
                    0xfbcdbe6ae1a4a4f8,
                    0x8ff8b996eba8b002,
                    0x2d0aa2fa600e2304,
                    0xa9c3c98875ad6d19,
                    0x642e1e7187c4b7f9,
                    0x0068c2b9cd1c2d6e,
                ])),
                Felt::new(BigInteger384([
                    0x0656f144edc01311,
                    0x4f4fd1091fb70340,
                    0x1de8c0ace101e5c7,
                    0x7fe4f15f2d39a0fd,
                    0x5181990e551cc3c3,
                    0x0141cb8f651e0b19,
                ])),
                Felt::new(BigInteger384([
                    0x8df58f7dbdb00a07,
                    0x9806eb822dad7885,
                    0xd022a05fa4d9a217,
                    0xe9bfc6ad8dbc4de5,
                    0x22a45aedcb86e82b,
                    0x007258bc6fbdb0c6,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6be1222b8abc92ef,
                    0x7790efbdb95e70f7,
                    0x09ba42fb6f14312b,
                    0x8b6a29ca4b6189b6,
                    0xae6c598c4feda382,
                    0x00525934c30b9cd6,
                ])),
                Felt::new(BigInteger384([
                    0x10c0fb6e120dfc41,
                    0x2c0fa52e3c923d3a,
                    0x6e0b69d52941ed4e,
                    0x1cf5d0a4cfc796c4,
                    0x2f8e5f04352f28e3,
                    0x012421d5100c0ff7,
                ])),
                Felt::new(BigInteger384([
                    0x007341b6237c9a08,
                    0x25512d879d87cd7f,
                    0xd964c89705865293,
                    0x9bc5cfab6d97933a,
                    0x7c8d71cdafa79898,
                    0x00251be92381f4d4,
                ])),
                Felt::new(BigInteger384([
                    0x08659f2959f5f548,
                    0x69fba7d572df5a93,
                    0x79c05c35dad80c62,
                    0x939953adeb62beff,
                    0x47068a39435c671a,
                    0x0007cc3615b1367a,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 4],
            [
                Felt::new(BigInteger384([
                    0x93b43ffffffff67b,
                    0xa0d125e30ffffb0d,
                    0x5d1a4faa05a59724,
                    0x323b39b7e2fcce8e,
                    0xf0223f35e4a1e060,
                    0x006f42bfb905f50e,
                ])),
                Felt::new(BigInteger384([
                    0x963abfffffff7099,
                    0x615462c0afffb57a,
                    0x983ad5e0f70bfb17,
                    0x043b91b1b7782f20,
                    0x460a32e63333859f,
                    0x005966855b430ccf,
                ])),
                Felt::new(BigInteger384([
                    0x93b43ffffffff67b,
                    0xa0d125e30ffffb0d,
                    0x5d1a4faa05a59724,
                    0x323b39b7e2fcce8e,
                    0xf0223f35e4a1e060,
                    0x006f42bfb905f50e,
                ])),
                Felt::new(BigInteger384([
                    0x963abfffffff7099,
                    0x615462c0afffb57a,
                    0x983ad5e0f70bfb17,
                    0x043b91b1b7782f20,
                    0x460a32e63333859f,
                    0x005966855b430ccf,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger384([
                    0x93b43ffffffff67b,
                    0xa0d125e30ffffb0d,
                    0x5d1a4faa05a59724,
                    0x323b39b7e2fcce8e,
                    0xf0223f35e4a1e060,
                    0x006f42bfb905f50e,
                ])),
                Felt::new(BigInteger384([
                    0x963abfffffff7099,
                    0x615462c0afffb57a,
                    0x983ad5e0f70bfb17,
                    0x043b91b1b7782f20,
                    0x460a32e63333859f,
                    0x005966855b430ccf,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x93b43ffffffff67b,
                    0xa0d125e30ffffb0d,
                    0x5d1a4faa05a59724,
                    0x323b39b7e2fcce8e,
                    0xf0223f35e4a1e060,
                    0x006f42bfb905f50e,
                ])),
                Felt::new(BigInteger384([
                    0x963abfffffff7099,
                    0x615462c0afffb57a,
                    0x983ad5e0f70bfb17,
                    0x043b91b1b7782f20,
                    0x460a32e63333859f,
                    0x005966855b430ccf,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger384([
                    0x388a5529952198b1,
                    0x5b16dc4d9943b485,
                    0xe3d6ef2dbb75bde4,
                    0xd59d9e5c00b9ed4c,
                    0x9ea26f11a9318375,
                    0x00503124f53e34b9,
                ])),
                Felt::new(BigInteger384([
                    0x41ed5fc008f308b3,
                    0xc825c0f02b5fda0b,
                    0x5d526c166c283a2b,
                    0x5cb8cb9b132afad5,
                    0xee1761926890a503,
                    0x001c2206b0ab2d2e,
                ])),
                Felt::new(BigInteger384([
                    0x4af6cca994fc10a8,
                    0x048ae3c76ccd0dc5,
                    0xb396f0b8dfa4c8a9,
                    0xe12414373fc94e86,
                    0x82e9bef9ffed4586,
                    0x00eec2f9ee9c9db2,
                ])),
                Felt::new(BigInteger384([
                    0xa8a2811eb9883f0a,
                    0xb6e7418dcac3995e,
                    0xfaca09512dde6513,
                    0x528f6eb219ed4a29,
                    0x7db7015dfed210eb,
                    0x00f71979d5279344,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb578e44fd7849a29,
                    0x8b26e87b92cf7120,
                    0x013b05645d915e83,
                    0x89ef8f70abf452da,
                    0x16c6a30f76bb5039,
                    0x0128ba108e5522a4,
                ])),
                Felt::new(BigInteger384([
                    0xfb0f72765d6a7bc4,
                    0xb7f93fd4061c29da,
                    0x62073df3b9215316,
                    0x1b9615b87830a880,
                    0xf7a8ee86a138ed29,
                    0x00e39f5c2144e327,
                ])),
                Felt::new(BigInteger384([
                    0x9dc5ec017703452f,
                    0x6057dc2aaec5b45c,
                    0x5a3cae87de5c0fa7,
                    0x2ada16480d4bb6b3,
                    0xd74510d72d7a0933,
                    0x000d958a0afae82d,
                ])),
                Felt::new(BigInteger384([
                    0x3eb2e96c861bf055,
                    0x80874ea2adc1a36d,
                    0x9eaab2dba4ba901d,
                    0x8c1ce7bb87812dc7,
                    0x4e67380e5c2345db,
                    0x018b0a935955aa2e,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe474bb1297c9541d,
                    0xb1802f9599530f5a,
                    0xf64df1b9ab451f7a,
                    0xba35f17c14db4682,
                    0x3006f8648fbc0486,
                    0x0088fdebf06f6bac,
                ])),
                Felt::new(BigInteger384([
                    0xb10f4676c29f4e7d,
                    0x98ba3cc0ff9a5253,
                    0x9979428652bea05a,
                    0x01ac35d71bd393a1,
                    0xcb9781ddfe277735,
                    0x009ba2e1e3cb291c,
                ])),
                Felt::new(BigInteger384([
                    0x9e01f21b934cd838,
                    0x0c2a757b02d8ab88,
                    0xa7e8354e9631744b,
                    0x104992a83620489f,
                    0x27c334f4f8b21b6b,
                    0x00e24e9b5faf19f8,
                ])),
                Felt::new(BigInteger384([
                    0x5b56b4c7c8fc025e,
                    0x807954e9477f4bb2,
                    0x5da21926c74af8dd,
                    0x18d767c697801ef8,
                    0x6c5d3ccd92860a3e,
                    0x01845815ca094388,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x8b9143a3370d1713,
                    0x5db649ce29f7cbf8,
                    0x008eebf319627ca0,
                    0x233cef0da93a98ac,
                    0x197b17269dba9c73,
                    0x0035b7881d0dcfa1,
                ])),
                Felt::new(BigInteger384([
                    0x086e14c660bd591d,
                    0x19e1b956bc5309a0,
                    0x7c350a652a790944,
                    0xdf8d8ceaa51b251d,
                    0x9e969563157d56fb,
                    0x015578f0a8a09bf9,
                ])),
                Felt::new(BigInteger384([
                    0x4890bdbafb6faca5,
                    0xd1f9929eda123858,
                    0xd535acd5eacfba60,
                    0x8d09e8d165aa5e27,
                    0xea620d8b3922bf05,
                    0x008a8cb4193f35bf,
                ])),
                Felt::new(BigInteger384([
                    0xf8f64436757f4c99,
                    0x7a37e90d68425ad9,
                    0xcd314eeb1eb882c2,
                    0xd8a20c993351289e,
                    0x9f5e417bbb7a3fff,
                    0x009fc4a7c30a0036,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7fe7e919f465734e,
                    0x03b45a4133e8cdb7,
                    0x3dee289fff55ddac,
                    0x53d8aa8b5818e3c6,
                    0x3863632732e68f3c,
                    0x010dfc3a9c40c75b,
                ])),
                Felt::new(BigInteger384([
                    0xcd15a6f033966681,
                    0xf823bc03464cbdbe,
                    0xb77090accbc29817,
                    0xa83c1e27968515ae,
                    0xb9edb9f9b19b896e,
                    0x011a7db21efa4385,
                ])),
                Felt::new(BigInteger384([
                    0x35ad7287abf127fb,
                    0x4037291cf966a940,
                    0x3c50b273d59000c4,
                    0x48aa8dd02994e594,
                    0x64b313801d483b16,
                    0x00d0c52057089d2b,
                ])),
                Felt::new(BigInteger384([
                    0x033aa73800e16abe,
                    0x5a304f9a3cbaedff,
                    0xaf0823f89427b146,
                    0xf0cc70f9954a7833,
                    0x06278f8ca74df038,
                    0x000b8643c076b94e,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x34d85d9e998e5ab4,
                    0x2609f8c965f00759,
                    0x46e3a09b959448bf,
                    0x3874dff46e809ba1,
                    0xb975b1472f642c82,
                    0x00a20df3c60de326,
                ])),
                Felt::new(BigInteger384([
                    0x0b39f7b911654cc7,
                    0xdc61096215a2ab71,
                    0xdbad85d490ba8080,
                    0xcefdd44543923fdc,
                    0x68119fad70460d21,
                    0x008d95791c3df9bb,
                ])),
                Felt::new(BigInteger384([
                    0x8a1db8d56e42fbbf,
                    0x82b1f5837dd46503,
                    0x17b4badf73ad9b01,
                    0x9a0da4c6564f4be3,
                    0xcd152f86218d0ed0,
                    0x0086349912897e02,
                ])),
                Felt::new(BigInteger384([
                    0x040e163799695a35,
                    0x71911b2b3cf9b8b2,
                    0xc12e32efe48d47a9,
                    0x3a070f7e7868b74c,
                    0x67df23a7f46651e5,
                    0x014947c8da7c1353,
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
