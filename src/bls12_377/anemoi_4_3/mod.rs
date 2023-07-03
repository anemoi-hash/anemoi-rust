//! Implementation of the Anemoi permutation

use super::{mul_by_generator, sbox, BigInteger384, Felt};
use crate::{Jive, Sponge};
use ark_ff::{Field, One, Zero};
use unroll::unroll_for_loops;

/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;

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

/// The number of rounds is set to 14 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 14;

// HELPER FUNCTIONS
// ================================================================================================

/// Applies the Anemoi S-Box on the current
/// hash state elements.
#[inline(always)]
pub(crate) fn apply_sbox_layer(state: &mut [Felt; STATE_WIDTH]) {
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
pub(crate) fn apply_linear_layer(state: &mut [Felt; STATE_WIDTH]) {
    state[0] += mul_by_generator(&state[1]);
    state[1] += mul_by_generator(&state[0]);

    state[3] += mul_by_generator(&state[2]);
    state[2] += mul_by_generator(&state[3]);
    state.swap(2, 3);

    // PHT layer
    state[2] += state[0];
    state[3] += state[1];

    state[0] += state[2];
    state[1] += state[3];
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

    apply_linear_layer(state)
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

    apply_linear_layer(state);
    apply_sbox_layer(state);
}

#[cfg(test)]
mod tests {
    use super::*;

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
            apply_sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
