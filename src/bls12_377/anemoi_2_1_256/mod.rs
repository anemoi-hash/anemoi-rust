use super::{sbox, BigInteger384, Felt};
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

    x.iter_mut().enumerate().for_each(|(i, t)| {
        let y2 = y[i].square();
        let beta_y2 = y2.double().double().double().double() - y2;
        *t -= beta_y2
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
        let beta_y2 = y2.double().double().double().double() - y2;
        *t += beta_y2 + sbox::DELTA
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
                    0x409d14f77f773dcb,
                    0xa62d21f921dc98b3,
                    0x737bc8034004baf3,
                    0xb8e35082ccb50e49,
                    0x6d1c3b6f57a1d1c8,
                    0x0172cb855435c4f4,
                ])),
                Felt::new(BigInteger384([
                    0x78377ba8a65a3d38,
                    0xca98a0756754b371,
                    0x8676f0bd65323c02,
                    0x26f614f8e8b671d8,
                    0x84150b8c256a7306,
                    0x00720f44ff79ecc7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc60cd47672f3c930,
                    0xc5f8460137985bae,
                    0xe78af7e7da0fa704,
                    0x3f9f10dd432d1e77,
                    0x9051f55e9223f279,
                    0x009a8088396d6e82,
                ])),
                Felt::new(BigInteger384([
                    0xf596857f1440d16c,
                    0x4eaf8e319ad48646,
                    0x4e062c4b7a8b883e,
                    0x12a56b2976bb68df,
                    0x1a2c67666cdc4de4,
                    0x0086b4204a6b9833,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xad80a29695921285,
                    0x782b0499df818c3a,
                    0x359c824fb1c2712e,
                    0x164016e7e1bee789,
                    0x5f50ce67108e803c,
                    0x00f06917a3e4364b,
                ])),
                Felt::new(BigInteger384([
                    0x79c436b66b482e2c,
                    0xe526ce1b7e66619f,
                    0xa0c749939da2e3c7,
                    0xce9a55a0436ada28,
                    0x123e29e2180c9fda,
                    0x015e0682d1e43417,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0d35ff7af3dd148b,
                    0x87d750e204a07d5b,
                    0x397bebf651846587,
                    0xb36a97eae5ad9e83,
                    0xa27b901a29392887,
                    0x0001fa6df4af5b1c,
                ])),
                Felt::new(BigInteger384([
                    0xcdd2f9e63fbcf324,
                    0x184249a2bd0fd3c6,
                    0xdf8ccf07dc603f18,
                    0xdaa21ac5b99e559c,
                    0xbbe7893e296f0061,
                    0x002d1c34349ad7b6,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7a5be15e6c29fefc,
                    0x2ed4a10c5b54a269,
                    0x8305153a3fba567a,
                    0x2a6b4ea643a38503,
                    0x128319f226174c76,
                    0x00f3be4239c1d1ed,
                ])),
                Felt::new(BigInteger384([
                    0x03372eb369545d7b,
                    0x1770912978099190,
                    0xcbd35fd81057f999,
                    0x4bc44b11230fd820,
                    0xd5986883aa591888,
                    0x00ff97763ef74bb4,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3f214de5e1749d9a,
                    0x8a16fb86d7bfb3ef,
                    0xbb105d7dcb47e1ab,
                    0x989b74b88cd717e2,
                    0x71eb825df5e22d59,
                    0x01423afa47cfccb9,
                ])),
                Felt::new(BigInteger384([
                    0x66d26fa9f78b73a2,
                    0x3850c8a3fbd611cb,
                    0x031cda55b5b79c81,
                    0x0734d5db8a5d5e1b,
                    0xad2a379dd5489d70,
                    0x01549edb6bfbc44a,
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
                    0x13fb4947c9dd194a,
                    0x5b95d047a400a365,
                    0xcfe6c034c95effe4,
                    0xa16dac2b28538927,
                    0x341bd6d9a7b370ad,
                    0x000a72f667abf4dc,
                ])),
                Felt::new(BigInteger384([
                    0xb8c04bc77aaf09da,
                    0x922856595695e099,
                    0x2f559b62cac41321,
                    0x6f44ba4814b0d7d9,
                    0xc528967f2e51cd63,
                    0x01341b76eacc48fe,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x60ad5d1504d3d8bc,
                    0x6488aad41a8420e1,
                    0xba0344919abf8d9e,
                    0xd4469f652a19cd1f,
                    0x2bd9e07aa7f42399,
                    0x01678d888cd69bda,
                ])),
                Felt::new(BigInteger384([
                    0x80156064a651b825,
                    0x0ae583f8b71c2f19,
                    0xafd2599ab6d822d8,
                    0x36c046170b0409eb,
                    0x64e3ccb2bb83e1f5,
                    0x0098d85292c7a554,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x732db143cc22b7d5,
                    0xd7e5cb149495cdd8,
                    0x1159c732e085370d,
                    0x67cdd163051dde5e,
                    0x26a93bbf2d0ea1eb,
                    0x01571694c11c6360,
                ])),
                Felt::new(BigInteger384([
                    0x2c9727c2aa0c6770,
                    0xee9e289ad84140fa,
                    0x3b209fa58b0c4596,
                    0x95c7d944ef09ca8c,
                    0x50357841ae826e5d,
                    0x0111745e90d1a857,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4dab80ca4d29b2c3,
                    0x23d5e2caafaebdf1,
                    0x6ad72e48b4e5dfb9,
                    0x4a2e357f80564573,
                    0x095b8ce0df63757c,
                    0x0079c23259626c8e,
                ])),
                Felt::new(BigInteger384([
                    0x1b95909b94626b38,
                    0xdba5ffbd66f4c805,
                    0xa708b92a92ed2e6a,
                    0x7cc881814c43aa54,
                    0xf0944bdf3bf4d6fe,
                    0x013cc2f1ae36894b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xec94e749bbe35e92,
                    0x169acf71e8c26deb,
                    0x379d0f5f0517f56c,
                    0x558ed74804762c07,
                    0x993c6e9789c6518b,
                    0x000761a9d8094184,
                ])),
                Felt::new(BigInteger384([
                    0x832edb9df4317b67,
                    0x46c82f9aa1e93781,
                    0x55cd7257eaf3a3eb,
                    0x88bf4e576a333e71,
                    0x313ac5c7161abe81,
                    0x002f81fb3245cf6c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x52a228b2be44b04e,
                    0x52c397e2ffad07d8,
                    0x538b816f3a7c69c4,
                    0x47be856a1fea2d38,
                    0x372bcbd9a6abf378,
                    0x0046e6728a8058e7,
                ])),
                Felt::new(BigInteger384([
                    0x8c24b10de0442c30,
                    0x641e3cb3d577e646,
                    0xd34849db31810bde,
                    0x27c2c06f8d050c08,
                    0x3ee07fa9b7c3a4e1,
                    0x01423b9dcebcb7f2,
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
                    0xbae5daf759e02659,
                    0xc7d6a509249ede5f,
                    0xbe93b79824d817a3,
                    0x763a3ca96a83e01b,
                    0xf2cef402d4ce13d9,
                    0x001bc61486a56cb6,
                ])),
                Felt::new(BigInteger384([
                    0x8bd1f94351d54db2,
                    0x6f27a79bc93a8ee6,
                    0x92c4b67be74bd1c0,
                    0x31cb873a5cfde503,
                    0x4c4a081d50578589,
                    0x008120276a047a1b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc5687fbeda3f9425,
                    0x94324e1329f6e80c,
                    0xee8958d63bdd3464,
                    0xdb806076819d4304,
                    0x13d33ffd90fc5702,
                    0x00557fb232c4cb2b,
                ])),
                Felt::new(BigInteger384([
                    0xd9ba729bcf27cbf4,
                    0x1c49f4d1ff452a32,
                    0xb3ba98c26663673b,
                    0x32a75fc5e072e655,
                    0x6a6aac90e17ee6c3,
                    0x001d25f4f5dea261,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x46c79320334c6890,
                    0x5ccedaa0a953360e,
                    0x72aa4a1e6d28a88d,
                    0x0f5221993240ec34,
                    0xfb3e7f70862e9dcc,
                    0x0163b4d21d166333,
                ])),
                Felt::new(BigInteger384([
                    0x967f36c03de802ad,
                    0x1a6e321229e793e8,
                    0xd0a1ebf0400ce8d6,
                    0x42e1c5bc970de1a2,
                    0x0d128a96dfada037,
                    0x010732bfec1db9a4,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xf293c419dc2d8dde,
                    0x3cb544cf9c00fbfc,
                    0xa25091e71a6e8c76,
                    0xc3028b5eff300666,
                    0x1d93ebc7208f01d7,
                    0x00eb79a5b24c575a,
                ])),
                Felt::new(BigInteger384([
                    0xc030d7da6519b59c,
                    0xc01405819ed361d9,
                    0xaccb992025dc8266,
                    0x7c885e0dff6f0742,
                    0x977494449297fe50,
                    0x009a2eb9a3017957,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x94275e6f58d202aa,
                    0x11a47e0b29bc3eb1,
                    0x9168c96a3c4e57f2,
                    0xe7e445bbfe2db9bb,
                    0xc141f5265cacfa27,
                    0x01a945ad8a99e5a2,
                ])),
                Felt::new(BigInteger384([
                    0x259cab5444c9fdab,
                    0x206949c052ad466c,
                    0x454f2a23144601c0,
                    0x6a0f5ae162a79812,
                    0x4994aeee72c32e60,
                    0x017ab4b78a94fa20,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9e9f65b077b73f77,
                    0x0b0d41d02e7c8402,
                    0x75672b901893cf7c,
                    0x707ee1f344de8e08,
                    0xc99d920bf4abbe83,
                    0x017846963a741f7d,
                ])),
                Felt::new(BigInteger384([
                    0xb4c4c4b57e92c09a,
                    0x655c4dca21163a4b,
                    0x5a90226086130393,
                    0xe9d093190eae5582,
                    0xec221c02368b9571,
                    0x000acb97f8091b90,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
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
            ],
            [
                Felt::new(BigInteger384([
                    0x15eefffffffff714,
                    0x669be3a3bffffb5d,
                    0xdc8ffe3035319f32,
                    0xd10f7bf375757f17,
                    0x6968af36d106a4b2,
                    0x019016a3edcd115f,
                ])),
                Felt::new(BigInteger384([
                    0x05547fffffff7986,
                    0x11c3dc611fffba1e,
                    0xda9e39e07be3a3e5,
                    0x4d4eefb142f7c397,
                    0xa2dc896fcece2a27,
                    0x00778a27853b0c5a,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger384([
                    0x15eefffffffff714,
                    0x669be3a3bffffb5d,
                    0xdc8ffe3035319f32,
                    0xd10f7bf375757f17,
                    0x6968af36d106a4b2,
                    0x019016a3edcd115f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd81075e9255fb3c3,
                    0xeefc02192f0d3ddf,
                    0xdc4ce01bca2441e9,
                    0xf89bc148d98ffd14,
                    0x523956b8d769c1f6,
                    0x00f3bf4b5dd450a5,
                ])),
                Felt::new(BigInteger384([
                    0x0c82e1ec8270d617,
                    0xb790dcf40b012f00,
                    0x83aac69eef216e74,
                    0xf1d50ae714c51cc3,
                    0xebcdeeee8a7f9928,
                    0x015783612b4cac74,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x024c76dffd948770,
                    0x257c491cef046107,
                    0x5784ea0a81a6f8db,
                    0xb92c231ba763ae7b,
                    0x89d856ba5aca9337,
                    0x005c7ec6830b3df5,
                ])),
                Felt::new(BigInteger384([
                    0x6d1b29bbaadbbb81,
                    0x097025b77086d99a,
                    0x77aa28d0d1102412,
                    0xbdd4e08bac6ae4e2,
                    0x2b66b03aed79ab55,
                    0x007de4c45c381104,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe5e44863d3e490a9,
                    0x02d225073de4e0a9,
                    0x76a54754e98d7d18,
                    0xf5303428027e6127,
                    0x010664c3630e24b9,
                    0x0001675405219aa5,
                ])),
                Felt::new(BigInteger384([
                    0x0edf7499a84c7c94,
                    0x44be5d7eca50bddd,
                    0xc45119e9ef573d3e,
                    0xa0b4d414bc7592f2,
                    0x1c727209ad81c71c,
                    0x011c40ac3915c94f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9c44a9e5c8af31fd,
                    0x0aa8c512fa63b7bc,
                    0x277d9fdab02bc87b,
                    0x8c51cc71f1e81183,
                    0x1e417f0998507a67,
                    0x018d1327c8891ee6,
                ])),
                Felt::new(BigInteger384([
                    0xa1bc4c51275da361,
                    0x1d5877e3aaab25df,
                    0x4bd898544beb519b,
                    0x476b6d711ea0fc1e,
                    0x8610b54e8e7d2926,
                    0x00571e3916435bfc,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x81dae85f60a7dfa1,
                    0xb530b69561e35f00,
                    0xefbce2dc3fe68232,
                    0xb0e2ada6b89890f9,
                    0x39bde499254ab092,
                    0x0052ac995c8da0b0,
                ])),
                Felt::new(BigInteger384([
                    0x325608eaeea01817,
                    0x7921e4b47effd772,
                    0xf4864e7ba4abcab8,
                    0x78eef9cd30b8dc09,
                    0x590402a65c3fab47,
                    0x014822e1af9231b3,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb11e2c52e250887c,
                    0xe46a7363eec9ee71,
                    0xa4e5cd0839a7bd1e,
                    0x0994a67820207d1c,
                    0xd962306cba3836f5,
                    0x006bfa37ab37ac10,
                ])),
                Felt::new(BigInteger384([
                    0x866f1d90c14abfdb,
                    0x8276f8d88eeb32f3,
                    0xa72cff4cb8cb4062,
                    0x2b1dc649edb66f82,
                    0x5631e11fd9f2f21c,
                    0x0153c608b8fcfdcd,
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
