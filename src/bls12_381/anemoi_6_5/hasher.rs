//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use super::{Jive, Sponge};

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// An Anemoi hash instantiation
pub struct AnemoiHash {
    state: [Felt; STATE_WIDTH],
    idx: usize,
}

impl Default for AnemoiHash {
    fn default() -> Self {
        Self {
            state: [Felt::zero(); STATE_WIDTH],
            idx: 0,
        }
    }
}

impl Sponge<Felt> for AnemoiHash {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 47 == 0 {
            bytes.len() / 47
        } else {
            bytes.len() / 47 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 48];
        for chunk in bytes.chunks(47) {
            if num_hashed + i < num_elements - 1 {
                buf[..47].copy_from_slice(chunk);
            } else {
                // If we are dealing with the last chunk, it may be smaller than 47 bytes long, so
                // we need to handle it slightly differently. We also append a byte set to 1 to the
                // end of the string if needed. This pads the string in such a way that adding
                // trailing zeros results in a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 48];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 47 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[i] += Felt::read(&buf[..]).unwrap();
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
                num_hashed += RATE_WIDTH;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        let sigma = if elems.len() % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        let mut i = 0;
        for &element in elems.iter() {
            state[i] += element;
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // 2*DIGEST_SIZE < RATE_SIZE so we can safely store
        // the digests into the rate registers at once
        state[0..DIGEST_SIZE].copy_from_slice(digests[0].as_elements());
        state[DIGEST_SIZE..2 * DIGEST_SIZE].copy_from_slice(digests[0].as_elements());

        // Apply internal Anemoi permutation
        apply_permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiHash {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);
        assert!(STATE_WIDTH % k == 0);
        assert!(k % 2 == 0);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = vec![Felt::zero(); STATE_WIDTH / k];
        let c = result.len();
        for (i, r) in result.iter_mut().enumerate() {
            for j in 0..k {
                *r += elems[i + c * j] + state[i + c * j];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger384([
                0x15a339f0a6af117e,
                0x235a007ccdfb9656,
                0x2d890c771d8e060f,
                0x72bfced488ba59e5,
                0x994155b03180ae67,
                0x0597a2d7a1855cf2,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x8daaa4dc43449c98,
                    0x6d215171c8ce87a6,
                    0xb92063416436f316,
                    0xd90666d4fe6cdafd,
                    0xc01ce4485f651dce,
                    0x07b7ad5130e5103e,
                ])),
                Felt::new(BigInteger384([
                    0xafac69473f276e96,
                    0x4d8ea9911b6d323d,
                    0x7438be4e38449f42,
                    0x7a19bbf27ad26a11,
                    0x0b6324f08d907eb8,
                    0x034dfb7b1775af59,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x708399b265260f36,
                    0x3b3d9db79e6f0fa0,
                    0x4363ce79806818bc,
                    0x6cb1a936572363c8,
                    0xb12a1e96b7059e74,
                    0x0b1e54ed96390bd1,
                ])),
                Felt::new(BigInteger384([
                    0x424fdbbab0b2888b,
                    0xdbbcb4be20a416b3,
                    0x92f065cbf04425aa,
                    0xa2c07c59582fec59,
                    0xb41a1063febc7616,
                    0x1433c1a41c0c46a7,
                ])),
                Felt::new(BigInteger384([
                    0xffd006ebc26d9167,
                    0x463286bdbfa6dcc7,
                    0xe5a8a2101d448880,
                    0xfd461509dbb336f5,
                    0x4969063352f54305,
                    0x10316cb24d4b65fd,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x62ab071ab98ad731,
                    0x8ab74ddf99cce8db,
                    0x4e33662998d96e5c,
                    0x9caef518cddd2ddf,
                    0x401ac08f9562f0a9,
                    0x132e768cc280ee47,
                ])),
                Felt::new(BigInteger384([
                    0x98d7db8af86208d7,
                    0x835907bdad29f490,
                    0xc211e2979ec74fb5,
                    0x68d8df3505b1f151,
                    0xfcb70d4a8738424b,
                    0x134a46864e22e3f3,
                ])),
                Felt::new(BigInteger384([
                    0x4e7f852918de09c8,
                    0x8c3e4b6876bc76d0,
                    0xc64caf5cd546cbc2,
                    0x97a06738ec6a8e62,
                    0x5f8661dbafe32359,
                    0x069c8dda59b15eb9,
                ])),
                Felt::new(BigInteger384([
                    0x664d0f502a0acb59,
                    0x140f9790725c1a30,
                    0x8cad30aa1bd2197d,
                    0xcd23ed904eb482f5,
                    0x4d1eb8d59125bd4a,
                    0x002e5eb0563e23d7,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x003e6cefb02d6d35,
                    0xbe94dea21c1c3695,
                    0x718c760672ab9772,
                    0x3810a58f04c9a102,
                    0x1d94edfeeeeb1f80,
                    0x0b497127c59fd6e8,
                ])),
                Felt::new(BigInteger384([
                    0x4be6cbae25b75436,
                    0xc6d3bc225ec29da7,
                    0x8fc3ae04f0c72400,
                    0x92a3cd76b7d2fa78,
                    0x0249f458db741925,
                    0x121bd6ebe0287694,
                ])),
                Felt::new(BigInteger384([
                    0xcb4ee70046883ad5,
                    0xfd9550f1d342ae94,
                    0x48bd0ee669378ec5,
                    0x5880287a744315fc,
                    0x1e818760ffb46a8a,
                    0x0bb23542bb19713b,
                ])),
                Felt::new(BigInteger384([
                    0xf42ac478473574cc,
                    0x8d54bb30e7412d8e,
                    0xedb9e53223c7266f,
                    0x516ca8cbf80579f9,
                    0x7814a83c89d026d6,
                    0x022d322b5acdd880,
                ])),
                Felt::new(BigInteger384([
                    0xfa6894e4df318abe,
                    0x45f7cc2d10cf675e,
                    0x14ff8d71652f354d,
                    0xc77e383d286aa840,
                    0x842f84e3a56f9074,
                    0x06932e7c76db0709,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x5555e0aca6b60147,
                    0xec7f8eb43c553e46,
                    0xae7b9ea1d78d64d0,
                    0x30015a0a6e8e6650,
                    0xc9fa548af3ffc04f,
                    0x193fab414ede6c46,
                ])),
                Felt::new(BigInteger384([
                    0xfdaa96629cc72430,
                    0x80856a65bd8af641,
                    0x4e010147ca27aa55,
                    0xba869b80e6de3e63,
                    0x835263f428cffddf,
                    0x0872099dcb28262d,
                ])),
                Felt::new(BigInteger384([
                    0x5981b530381933b4,
                    0x730f5354f497ddcd,
                    0x605f14d85db15bd0,
                    0xcfb66b81a3e92293,
                    0xf51ee0f5f0182562,
                    0x0944f56332143bb1,
                ])),
                Felt::new(BigInteger384([
                    0x04713fce2dd20eec,
                    0xc8d21387827dae8e,
                    0xfee33db0687b5d11,
                    0x42ad82a580d5cb16,
                    0xbed4e64ba92631fe,
                    0x16c01d99c2e41684,
                ])),
                Felt::new(BigInteger384([
                    0xd4a1087b54b0cf64,
                    0xa91cef882dac31e1,
                    0x7616aea014efe7d2,
                    0x4dcecfb1b7cd9ee2,
                    0x03124360c602407e,
                    0x00fa24ca685c61b5,
                ])),
                Felt::new(BigInteger384([
                    0x7a3f467c674d8b1c,
                    0xb09b897239f4b5c9,
                    0x9577017e27f49af1,
                    0x86964c8d897749e3,
                    0xa3ec1d60a85d05b6,
                    0x0d0610dc811a6d13,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x9430af7b689df45f,
                0xa0c7688557ad643f,
                0x430a8c7eedc09972,
                0x46f073a38ab2b081,
                0xaa917b45c5c86bd7,
                0x0734f64375f3e05d,
            ]))],
            [Felt::new(BigInteger384([
                0x767e30e1823fbcc0,
                0x61bff333a2c2fdb6,
                0x76604283247a6191,
                0xb1ae042b727dca78,
                0x91bc4eb58e47d410,
                0x19c24eeba0764729,
            ]))],
            [Felt::new(BigInteger384([
                0x3880be6e0e69286a,
                0x41d239a50e2c41e3,
                0x0176e70ba8b30bcc,
                0xfe3cc2a2cf76d6bb,
                0xfc68506e283cfecb,
                0x0fe328465da1bcff,
            ]))],
            [Felt::new(BigInteger384([
                0xbc57b40745401539,
                0xa89b16ef805342a9,
                0x76adabebd5dc6c9b,
                0xb056c75bcddb7b80,
                0x9366953396234b7c,
                0x1647b073302af426,
            ]))],
            [Felt::new(BigInteger384([
                0x737a1043f230b8ee,
                0xb0505ead5637c29b,
                0x25b971b4369cb557,
                0x80e11527e3ef36e7,
                0xed7ffea7e6c58acd,
                0x0cb0d3defd31bf21,
            ]))],
            [Felt::new(BigInteger384([
                0x1544f1ac7d105413,
                0x193d0ef6deaf76d4,
                0x8b644a4eb4fbf77e,
                0x887f739ca64b2d0e,
                0x4df5a6a845ca671d,
                0x0b7c58746eac1a94,
            ]))],
            [Felt::new(BigInteger384([
                0x0410fd2a0337d00d,
                0x804e3744e19527b4,
                0xdcdb363bf130887e,
                0x31ec577e834a0647,
                0x0d1c12fbca26f9ea,
                0x0576c8e088488dab,
            ]))],
            [Felt::new(BigInteger384([
                0xaaff737b8f3b4a7b,
                0xcecca1eea9f0fde2,
                0xf0a178b6fd8c5f87,
                0xb20db89dd1e5300b,
                0x424517816940d430,
                0x1666ff6873aed248,
            ]))],
            [Felt::new(BigInteger384([
                0x7e65d7db2f28441e,
                0xa2b44290140a62a6,
                0xefacd9b727f94838,
                0x52efa61d03e1f471,
                0xfe55d5c023e743c6,
                0x0e6d1ab07d6fbb57,
            ]))],
            [Felt::new(BigInteger384([
                0xac42f7715cc7b070,
                0x73a27b865b9facc3,
                0x7ee94277fba8c61c,
                0xe167348e4753aa2b,
                0x88ba437f80a0bd5b,
                0x0c6c64fab8073718,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x4e660ec470af4b23,
                    0xf25f3448d2bacd47,
                    0xd71d7702e7268632,
                    0x07d76fae60bdb075,
                    0x21ec993e4b038704,
                    0x044ab74a4e5425e2,
                ])),
                Felt::new(BigInteger384([
                    0x61c4bf4c3a244e79,
                    0x5476ea303645f11a,
                    0xb6cc5e42eb211678,
                    0xdefdfd680bd8aec0,
                    0xe2dd65068022ce81,
                    0x0e52b8a426e254c8,
                ])),
                Felt::new(BigInteger384([
                    0x4f7c6fa78c50f6ec,
                    0xad4ca411d625e179,
                    0xa5d714c692300d1a,
                    0x22ebb562a78a374d,
                    0x265d195f29717060,
                    0x13c8d1d2e7e38b2f,
                ])),
                Felt::new(BigInteger384([
                    0xa78bf1547d30fced,
                    0x523743f0fe333655,
                    0xf9b3c781b53e537e,
                    0x3e1201d9af032b88,
                    0xc6c0ac966dc31a1d,
                    0x07061df02be8d8b4,
                ])),
                Felt::new(BigInteger384([
                    0x1497529eafd05716,
                    0x86d4cb2d3b939e9e,
                    0x34e137d32561104a,
                    0x5287be99df9d80f3,
                    0xd4430abf7110d4eb,
                    0x08df7e5b38cbc52a,
                ])),
                Felt::new(BigInteger384([
                    0xbcd80324be485ba9,
                    0xf0c9c573b3084b43,
                    0xd407bba6592cadb2,
                    0x4868a8e018489a03,
                    0xe7ff9ab8df5951d2,
                    0x013444ec7542e5b4,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xe98303e2f1f990d8,
                    0x0d5d4ca458e015e4,
                    0x1ad8b0abd82b3f8e,
                    0x4502667499417e1a,
                    0x8d2388a5d1e881a5,
                    0x0c8f9b9c0793baeb,
                ])),
                Felt::new(BigInteger384([
                    0x05d8d935b065f297,
                    0x8ee4eedc132cefed,
                    0x9f36dd7a299d42f1,
                    0xfade3c864f5712a7,
                    0x9e84b30440d796a6,
                    0x0fb93616c84f691b,
                ])),
                Felt::new(BigInteger384([
                    0x5b4957739460e530,
                    0x976f9684ea11835f,
                    0xe78d459bd1c4b95a,
                    0x61d403f91fc89089,
                    0xb610305fb0544a80,
                    0x0399f0d21b83522e,
                ])),
                Felt::new(BigInteger384([
                    0x3c2efd1766260470,
                    0xca9cf2f0060c0982,
                    0xb53959e356096841,
                    0xb854e120f6c69931,
                    0x490d06390f71425c,
                    0x0d6aee4a866aca96,
                ])),
                Felt::new(BigInteger384([
                    0x7c529aff284c4694,
                    0xdecfbfc54fff9a3f,
                    0x18bd89e52f07d0d8,
                    0x6653e15e2fc64ecc,
                    0x8090c974ea6e31c8,
                    0x031310a867e06ab8,
                ])),
                Felt::new(BigInteger384([
                    0x904f769cfc9fa4bc,
                    0x84ca8d0f0a3ba6af,
                    0x0da34fa5801acac3,
                    0x51f63772023fc3a9,
                    0x48399c05a7f7b48c,
                    0x07c52224d93c8a06,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x5ad0811c26cdd283,
                    0x06e4c72716be6125,
                    0x317eb436a742c4cd,
                    0xfc97a03e9e7c138b,
                    0xe401072526f9884c,
                    0x0b916285dc019573,
                ])),
                Felt::new(BigInteger384([
                    0x99fce298078e0e66,
                    0x84d7a0dfb88582f1,
                    0x0668ae40447d8c0c,
                    0xbd853ffed6cf5ea2,
                    0x0694d16c7b2b8105,
                    0x0f97e2a51162ddc1,
                ])),
                Felt::new(BigInteger384([
                    0x44169920da96c886,
                    0xfd6e2cc05792e852,
                    0x128a1a6ae22b4c6e,
                    0xac3de36da43c6f6c,
                    0xefc8c946420f45bd,
                    0x0d7f7d8911529ef5,
                ])),
                Felt::new(BigInteger384([
                    0xd538e532e588b8e4,
                    0x116c16a8d85a611f,
                    0xd7ec9720efa5ec77,
                    0x37a3bd8c56a04c38,
                    0xb014c9ad85dbe0fd,
                    0x003d8db721d8afd1,
                ])),
                Felt::new(BigInteger384([
                    0xb0648627555fa4e0,
                    0x28ea7a8152600d38,
                    0x2a012b4fdba31bc8,
                    0xc803884931be5a3e,
                    0xf69b3806a8eadd81,
                    0x163dbde57c036ce4,
                ])),
                Felt::new(BigInteger384([
                    0x88c3a07b742db609,
                    0xe89ec49dd0b1b662,
                    0x7e24928a0127fef7,
                    0x510f0f7e78b98679,
                    0xfb9a087e3e38ca0b,
                    0x175ffb3268ab90a5,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x9d65d39cbc080e06,
                    0x57fe128cf14802df,
                    0x13b80343142ef94e,
                    0x5fd17c151db58c49,
                    0x841270332cdb5d4d,
                    0x0da2da633c47a7d3,
                ])),
                Felt::new(BigInteger384([
                    0xad2a092aaa9da1c9,
                    0x2b44a668f3cd93ff,
                    0xcae103bdfca037d0,
                    0xdf39d939f077ded2,
                    0xd47457dbee81d9c2,
                    0x112b883f293b8a99,
                ])),
                Felt::new(BigInteger384([
                    0xaa9ff77254957b9e,
                    0x80f7ec7180d5ccd8,
                    0x6ee7c171134ad454,
                    0x1d57cb857d04c9eb,
                    0x328ec17999e83dac,
                    0x0683b5b3591c5123,
                ])),
                Felt::new(BigInteger384([
                    0xdb8ea94d543b4e94,
                    0x0d510f46295daef6,
                    0x81a6862fb14786b4,
                    0x991ff49b7ffa43a0,
                    0x9029e8d2341fa121,
                    0x1350144bb7d482c0,
                ])),
                Felt::new(BigInteger384([
                    0x2e8e64a81c018920,
                    0xa06addf1f7b1cd0d,
                    0x9f0113bf20ffea66,
                    0xf68cdd66d35ca7c5,
                    0x2ddb50709f438104,
                    0x0179188ecbdba9ee,
                ])),
                Felt::new(BigInteger384([
                    0xe2db2e6f1de084ca,
                    0xf8ca2ed57cd992c7,
                    0x4feb2dc91009d818,
                    0x8f9561a014e8a8b7,
                    0x041ec8f931bfe631,
                    0x19a772c3b8b3c3df,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x308da6187e600148,
                    0x02ed9608caab997e,
                    0x3963e0bbb815dc2e,
                    0x7e44c8b71ba1938f,
                    0xa5cbbb817ee6dbc8,
                    0x0454846e004f9cae,
                ])),
                Felt::new(BigInteger384([
                    0x986dba21eec7e72a,
                    0x0b3e85469f0bfee1,
                    0xc104c200b02f1ef8,
                    0xe93777f24be05db4,
                    0xd2443a625b0a95a9,
                    0x17c32ac951b61a66,
                ])),
                Felt::new(BigInteger384([
                    0xfe34fe262643ef3d,
                    0x21c675cc8fa2689d,
                    0x08507a0f6cd85b5d,
                    0xfb8faf2f214f8273,
                    0x32fe7cd150b82340,
                    0x0e54e01bd5ac88d6,
                ])),
                Felt::new(BigInteger384([
                    0x0ddf5079f1e752eb,
                    0x4d73fbe2e391f0e0,
                    0x6804001f54492a81,
                    0x4cfca9fbefe7fe9c,
                    0x6536765131bb293d,
                    0x1221c48837a4c2b8,
                ])),
                Felt::new(BigInteger384([
                    0xe3993cbccb6fb427,
                    0x404c7913d9684668,
                    0x7235bff7a824b631,
                    0xca982475701dc663,
                    0x1d68567c5eb2302e,
                    0x150f1d007745a4d6,
                ])),
                Felt::new(BigInteger384([
                    0xcb2ebac0fbde8653,
                    0x04eddb322741a5dc,
                    0x595ebadad3a222d9,
                    0xd551ee6cfe5c61ef,
                    0x30db6aca8220071b,
                    0x02d60e8a8106dc05,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xba99b6216e0ee438,
                    0x6bd0a1ebb94bd578,
                    0x8faace46360332af,
                    0xc123b038340e8857,
                    0xf4e81136c18df1ca,
                    0x184305473c659073,
                ])),
                Felt::new(BigInteger384([
                    0xece7480f51dcf02f,
                    0xcb2bf854c9c705e1,
                    0x572e63c5a81c2600,
                    0xceea8b93ec98eb26,
                    0x82be27c26281dbe7,
                    0x0deb0c3913c4d49d,
                ])),
                Felt::new(BigInteger384([
                    0xf7fa490d7618e80b,
                    0x6ca92d3f69e5d0dd,
                    0xa3b3efb06104712a,
                    0x2bbe354716b6a8f8,
                    0x1009f392cbf49b0d,
                    0x15cb987b7f299868,
                ])),
                Felt::new(BigInteger384([
                    0x530bff7f2e9335ce,
                    0xeb237d0d40417926,
                    0x140f5412bd69b636,
                    0xd34e48c1e4699821,
                    0xee5feb4eaa8c6bdb,
                    0x0146996ebf83b9a9,
                ])),
                Felt::new(BigInteger384([
                    0x1a8816bf1ac35ba1,
                    0xd0bc6d711f7c92dd,
                    0x284dfe8f824b287c,
                    0xe643595a8273a330,
                    0x9bb18959a9c9eb6e,
                    0x08b25b4c78bb7e28,
                ])),
                Felt::new(BigInteger384([
                    0xa41ac17c7ae536b1,
                    0xba5de2724bd31409,
                    0x7434c0ac92f78ffc,
                    0x269b6219b7ad438a,
                    0x01b8ae9fba1efeea,
                    0x1632c27188fc3b40,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0x569cf0368d2dc73e,
                    0x3a439e598ccdbb35,
                    0x7997989fd9690f55,
                    0x21631ba6e67b21a6,
                    0xa6f6e936022959ac,
                    0x192464a359bd988d,
                ])),
                Felt::new(BigInteger384([
                    0xb0d0a843fe46c5aa,
                    0xace2191cc5460986,
                    0x576fc8897d6c9e78,
                    0x4701c5183ed739c3,
                    0x2126e5bcb53ce3d1,
                    0x10260a1a4d03ed6c,
                ])),
                Felt::new(BigInteger384([
                    0xc05f56a8368abf63,
                    0x7249a1ef4c36e7f6,
                    0xe59f893fa9bdc14e,
                    0x929b35b84f19c9cb,
                    0x8d17a2e07582725d,
                    0x097113343ad626a8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd669750ca640187d,
                    0x4dbc99d9e8437c97,
                    0x453a42cfed67298b,
                    0x0711d59ed45c32cb,
                    0x11f60e24dcfad3b4,
                    0x01d0dd9666ce7973,
                ])),
                Felt::new(BigInteger384([
                    0x221d3e71b3ef1921,
                    0xf332f0a2c9c29771,
                    0xf233cdb534126e87,
                    0xf4265e50af2b9dcb,
                    0xb5352ef742b884b5,
                    0x01573be474289678,
                ])),
                Felt::new(BigInteger384([
                    0x0d0c4198d7b0d782,
                    0x3fb5d0ed744b30e8,
                    0x77de0d7911a903dc,
                    0x2278e6d8b236ae18,
                    0x78dc1b8c6271e40c,
                    0x0808085bdd9d71f9,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xa015314f90efdb1b,
                    0x50c2a1a5309435b9,
                    0xfa2219ff5be8a3a1,
                    0xdd9d04fb18cb3e37,
                    0x16ba943b9562f156,
                    0x0c5477d247f85d59,
                ])),
                Felt::new(BigInteger384([
                    0xa3d3739772d1e0d3,
                    0x31112f39689c80d6,
                    0x448cba0a0934577a,
                    0x087cc79b39b615f4,
                    0x0f649ec834120f79,
                    0x0e36df1d7d23c69e,
                ])),
                Felt::new(BigInteger384([
                    0x59de797bb33f20c5,
                    0xa270474b2a781243,
                    0xef5c59283c795d22,
                    0x3b408fde8e2c8930,
                    0x274ec4fa7a17bc35,
                    0x14431b1b695100f1,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb710ed995092ddbf,
                    0x3492e2b30990ea40,
                    0xfddf4bb9b6054781,
                    0xa87480d4615002dc,
                    0x38e9e9a5d6dea591,
                    0x15231ddd1bb1da53,
                ])),
                Felt::new(BigInteger384([
                    0x91f7ed0f38222997,
                    0x102033d3133fa20e,
                    0x8f051d5b167a3441,
                    0x6351a9bdca1f0b40,
                    0xfa80f2481c6d07d5,
                    0x086df95c03cf0c78,
                ])),
                Felt::new(BigInteger384([
                    0x8f2be78d4c9df4c5,
                    0xe0c847fbe8f16484,
                    0xc615c6ba43ab8f8a,
                    0x6b46925d1da5e721,
                    0x6c919cdf9f6c9d76,
                    0x170938a47e7a1e19,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x16291fad980b5c80,
                    0x910d2619f621bf09,
                    0xdfdfd434f3540e7f,
                    0x630b65a1fabfedbe,
                    0x0e2eb1be09274e5f,
                    0x18ea3522e533f8f6,
                ])),
                Felt::new(BigInteger384([
                    0x483e4c1c8ea987b8,
                    0x6730987138644b25,
                    0x5a1af112a7972915,
                    0x5a0e01ddd014c18e,
                    0x09276edb23b81071,
                    0x1479fb232fb75840,
                ])),
                Felt::new(BigInteger384([
                    0xd135ccb466ccafea,
                    0xcc578b34b6e2db95,
                    0xb0089b2afe991f66,
                    0x15e8f99c6681c249,
                    0x5c231afb126864ca,
                    0x019bad41cde2e157,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x506aec58b79e3024,
                    0x3f2daddc09eb2eb8,
                    0x358fc5460601510c,
                    0xc6040e6e3a72789b,
                    0xe403a2fe92bf7357,
                    0x0d348c3a4f963af1,
                ])),
                Felt::new(BigInteger384([
                    0xf4bf167f98a95523,
                    0xb1062e5f74c73995,
                    0x297b75038bec686b,
                    0xe0b8cc4c707b5932,
                    0x97e42116e7baa33a,
                    0x12c0264c8bf45184,
                ])),
                Felt::new(BigInteger384([
                    0xb9440d12a2f30d2c,
                    0x31c0e5045417278c,
                    0x3dd05600f12348f2,
                    0x8d34290a0c7f1353,
                    0x45b20f3152fb598b,
                    0x103592495e3f734a,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x451b38b428509719,
                    0x4688080ce175dfe9,
                    0x574edceae6a6c3e5,
                    0x9feabbb3177ecb82,
                    0x6f2bf6ed114827cb,
                    0x03d3b20bd2a1a47c,
                ])),
                Felt::new(BigInteger384([
                    0x0e623bbfa02b9ccf,
                    0xfe8f1d0eab84ca48,
                    0xca7a2a42aa8a5f41,
                    0xdfbe1005d697d96e,
                    0xd79e865a90510b7f,
                    0x134b6560435ad9e6,
                ])),
                Felt::new(BigInteger384([
                    0xf24642d8f772d763,
                    0xa5ee26395b81fc5a,
                    0x2ee246397e699e4d,
                    0x59bad5fe704186c6,
                    0x007a83302f580c74,
                    0x06bb8538d987335c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x63be3192ebae9e08,
                    0xf6864f2b7eacea99,
                    0x15854fa696cfad41,
                    0xa0bcde23203abd04,
                    0xb30024721d0b749a,
                    0x113845d7244c090e,
                ])),
                Felt::new(BigInteger384([
                    0x55ed169c51a0d105,
                    0x5a5bd20fbded782d,
                    0xcc83fc8748c401ed,
                    0xc2fb35be496c08d4,
                    0x8318adc8edd3af19,
                    0x04aa317ed7a2e698,
                ])),
                Felt::new(BigInteger384([
                    0x2caab50237a16c67,
                    0x07bf0823225f74c2,
                    0x80c30897cb26154c,
                    0x4f0475837f2da4ab,
                    0xb29f876b5fa246e1,
                    0x0006c5a8a3aff8a2,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4b7c71afac855471,
                    0x57f274a13912a2dd,
                    0x34b0b7e4533ceeec,
                    0x9e840ed1f4d790f5,
                    0xb1bdea1ca096d258,
                    0x071e21d2944b05b6,
                ])),
                Felt::new(BigInteger384([
                    0x090afa0d2f4cdfa0,
                    0x5b2174a30d7a6993,
                    0xdbf92d9240306ddf,
                    0xa56289333f133678,
                    0x7c3e2caa2a3141d7,
                    0x0a5f62a8218cde5d,
                ])),
                Felt::new(BigInteger384([
                    0x3f174d5579d1121b,
                    0xe352fdd61277a4e8,
                    0x995a777b5c40c181,
                    0xf6518b2279950c84,
                    0xdaef2e005608a058,
                    0x064d315f80400858,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4f8393124c410216,
                    0x0dc351808090e512,
                    0x707e6122c2299806,
                    0x2b04e589149811e7,
                    0xc1dae24ba9ba2493,
                    0x17610892ec5ecd26,
                ])),
                Felt::new(BigInteger384([
                    0x739b71b0d2f95fa3,
                    0x2c0e63ee0fa8eaed,
                    0x66e8d1058c242d6e,
                    0xb96c861a24096f65,
                    0x1cdd72851530439f,
                    0x0975e20795fbb5bb,
                ])),
                Felt::new(BigInteger384([
                    0x8e552c6d34e0c5ac,
                    0x9d49b8d90d1d3401,
                    0xce6de3639dbb196a,
                    0x1f3cfbe1036f65ce,
                    0xe6a7b99f0464696f,
                    0x03ae9450e0fb05e0,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa9d9be60a6f713de,
                    0x69dbac9500e78413,
                    0x09807ec184c6d966,
                    0xecd3695675e1c65b,
                    0x6aa5e6cc143fde38,
                    0x1632411ad94c7085,
                ])),
                Felt::new(BigInteger384([
                    0x860514a941700503,
                    0x7c0395779baf4fd8,
                    0x9d5e4efdca4ce405,
                    0x661cd3582df89f78,
                    0x76472ccc249416d1,
                    0x08ca9c1b8574f6f7,
                ])),
                Felt::new(BigInteger384([
                    0x9008118b75a5d9c5,
                    0x80e054801ef1eb9d,
                    0xc28eb4bd76b6a452,
                    0x69a669c0eae33b11,
                    0x792b431cc57fa7e5,
                    0x0222311fd5f60ce5,
                ])),
                Felt::new(BigInteger384([
                    0xfb2ce74e6eb4f1d5,
                    0xa16a6589b46f861a,
                    0x5f19a5dc2cf687a5,
                    0xb7221e024aead1e8,
                    0xb2ab34c780942672,
                    0x119d4d8a6f4b07d4,
                ])),
                Felt::new(BigInteger384([
                    0x8f7677e8631e3bb5,
                    0xadae3e8e7a4bae39,
                    0x2b46b48da80848dc,
                    0x4ecadd6af3a4fc62,
                    0xffe0311263cc7796,
                    0x0355bb94ef8fa049,
                ])),
                Felt::new(BigInteger384([
                    0x6bf2641843df4421,
                    0x5c974b36810db3a0,
                    0x9ac927d77e05296d,
                    0x7e7c73c36b0a620e,
                    0xb4b9caa7d50ed205,
                    0x00a65a41bb2a1067,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xf4cf22aadd9f8762,
                    0xc21cd76a3add88b8,
                    0xa739a4af78e42040,
                    0x3703211b9ca357e7,
                    0xdc74c8f59c5d103e,
                    0x0d7229aedddf746f,
                ])),
                Felt::new(BigInteger384([
                    0x6846e4afd1c8a815,
                    0x3b92f56aac67202e,
                    0x89151e7ab9486201,
                    0x3cf27d77d8697056,
                    0x3755ba2c1ce61f25,
                    0x186ce7057cf8f752,
                ])),
                Felt::new(BigInteger384([
                    0x2b0fdd1ce74fbbd9,
                    0x49ac9de43e38b170,
                    0x81601c2d5c4b7485,
                    0x5e48a2e22b8441be,
                    0xface641ea78cc4e5,
                    0x1385e79e655f8ea3,
                ])),
                Felt::new(BigInteger384([
                    0x8c8dcb38aaeae98c,
                    0x6d8a4f65df7ac239,
                    0x642a3764c2c13428,
                    0xfcfbb8da53286f81,
                    0xea309c749e0d8933,
                    0x13eab120d1d2b881,
                ])),
                Felt::new(BigInteger384([
                    0xf1f95203e18f1239,
                    0x6d04116c0e3d4acf,
                    0xbca0a26a8943662e,
                    0xadef492497bba894,
                    0xefef8ae08dc91b9e,
                    0x146210317da47be1,
                ])),
                Felt::new(BigInteger384([
                    0xcbd5fdeaa25b08cb,
                    0xbcaf3e362f942adf,
                    0x0ad29b064ff1ee8e,
                    0xb05db7b7c0c8d2fd,
                    0x02b9eba1a530dc0e,
                    0x102ad5b3569ff844,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xb6dca05026246b1f,
                    0x3590cacb2ff2fb86,
                    0x1a4c3c35a25ec191,
                    0x602bec23fd5c6f71,
                    0xa6871e72dc07856d,
                    0x18bd85ca1ae48ef0,
                ])),
                Felt::new(BigInteger384([
                    0xa8a52223414a2805,
                    0x8480ebcdf8feefef,
                    0x9dd7515a347bc3b1,
                    0x4dd156ca85544b9e,
                    0x824ea6dd3ea1ce89,
                    0x05f8b79433d6a667,
                ])),
                Felt::new(BigInteger384([
                    0xeed2ccb5f016380b,
                    0x8aed8091f7b4d61a,
                    0x5af59006b99d23b1,
                    0xcb56b9cb6f676aac,
                    0x33d1be624b09f26d,
                    0x0b82c98eb57a0f4b,
                ])),
                Felt::new(BigInteger384([
                    0x846eef17eead14c7,
                    0x53e10da3e8bc1638,
                    0xf9ac764792b1e594,
                    0x99b6ffbd1292f377,
                    0xd4d12e9c47170380,
                    0x0ebf40bc14b5391e,
                ])),
                Felt::new(BigInteger384([
                    0xd40a1895f9ddcef1,
                    0x62489a50b004c641,
                    0xf3f1c6f23f35ebc3,
                    0x32933a28df82838d,
                    0x0aac44a65a724363,
                    0x08cf08bffea0fae5,
                ])),
                Felt::new(BigInteger384([
                    0xdb3015df8ddfc370,
                    0x50a88f461a4a9d9d,
                    0xdf4ef198734c77d0,
                    0xc7eacd4e829d52c0,
                    0x3fbb2f48b2b37997,
                    0x0339d7d36e8b6def,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xf667324bcf953396,
                    0x58532bffe8cf5218,
                    0x43cd2a25310cdc7b,
                    0xd8dd1da2c16d7075,
                    0x7403dd8d1a5f232a,
                    0x1082e8b8a7bf08bc,
                ])),
                Felt::new(BigInteger384([
                    0x85c57d9a956b1030,
                    0x72df9baec8964b62,
                    0x09edc793ba273cf0,
                    0x4ec164c8db826a5a,
                    0x593eb1d6b31a46f8,
                    0x1009248b452bb13d,
                ])),
                Felt::new(BigInteger384([
                    0x79bee28dac8e14cb,
                    0x9de8b537b11569c7,
                    0x644dc2bdc13949db,
                    0x5342ca50ec1956f7,
                    0x7747c3d959f4e4b9,
                    0x12cd59fc4a98a03d,
                ])),
                Felt::new(BigInteger384([
                    0x29f6e0edd103fdfa,
                    0x8db175c88b2d5bec,
                    0x7608542fea4e4ea3,
                    0xc066ee61ba000a55,
                    0x08e2790bd3d4a020,
                    0x0abf62ab72926536,
                ])),
                Felt::new(BigInteger384([
                    0x0ec0a1e16e0af10c,
                    0x71e1170b94afdd7a,
                    0xd431a280fe487846,
                    0x027ba3b85a9ab0c3,
                    0x4d004ebfdb41bfeb,
                    0x0674d0e704474b61,
                ])),
                Felt::new(BigInteger384([
                    0x7895c8bbdcef6739,
                    0x012af3f3c36f752f,
                    0x83abb2eef71b2125,
                    0x607fd7ae6c835600,
                    0xb4db00c113ab3081,
                    0x115f85aa4e6d8a03,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xaae6e7843b416afa,
                    0x949a3a1218d96006,
                    0x51b52dfcba1fca29,
                    0xcfe1c0d0bc83752f,
                    0x809068e747bc03e3,
                    0x18dcb78bf2733fe7,
                ])),
                Felt::new(BigInteger384([
                    0xdaa37331abe17ce7,
                    0xaccc587e9aff4086,
                    0xf1038fc2104c5c09,
                    0x3f2a1f01fd01fca2,
                    0xe350efecf610db6e,
                    0x1074772c9614ab2a,
                ])),
                Felt::new(BigInteger384([
                    0xdeff6422504de7ae,
                    0x3ca5d3da8a3888d8,
                    0xe4bbb33a8da9a4b6,
                    0x2b61942332b1a172,
                    0x423646eaac06110e,
                    0x00d22303c4281374,
                ])),
                Felt::new(BigInteger384([
                    0x17198a2b7b73824a,
                    0x2a74feb1484600b7,
                    0x648024d0c1119216,
                    0xca08acebb5657516,
                    0xb5018a19220f6f5b,
                    0x0aeb424cdb7ca9be,
                ])),
                Felt::new(BigInteger384([
                    0xfc2f4548ca18b56b,
                    0xc4fbd7eaa105f019,
                    0x82bee21c90f75d24,
                    0x59933aaebb87d6c4,
                    0x4fdedf8ccb51ccfa,
                    0x1245cb2e7725da74,
                ])),
                Felt::new(BigInteger384([
                    0xd6e2239a4dafac37,
                    0x8df93ec32cb0b5c8,
                    0x5fb2bff0e5de6035,
                    0x9ba368cd63464ad5,
                    0x22efc2b321f7e25f,
                    0x0380cb2608320eae,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xe33fcce2c4b60213,
                    0x84517758e97c6e59,
                    0x2021033f2e9db795,
                    0xbe14fdf13373d683,
                    0xc283b91af245fc77,
                    0x19c8baf631a48c30,
                ])),
                Felt::new(BigInteger384([
                    0xd6e137cfaa1c6440,
                    0x743a124549f3562a,
                    0x1e6d13b0f262e2ad,
                    0xe7a96dfa6ae4b05b,
                    0xf618b9354b93e09c,
                    0x0b7bc72274bd0be9,
                ])),
                Felt::new(BigInteger384([
                    0x61902659a3530ae2,
                    0x3a671f1efecd00e9,
                    0x692dc561a5a4e9c9,
                    0xcf6099afc799b6bf,
                    0x8ce2c49475985fb3,
                    0x10b97ce26243a878,
                ])),
                Felt::new(BigInteger384([
                    0xa15e4fc65c509422,
                    0x8849a19633ff08a7,
                    0xf13c826670b512e0,
                    0x3d1ef9ca961d9ba6,
                    0xb2f0df09e2f622bb,
                    0x01168db7b0ef49c5,
                ])),
                Felt::new(BigInteger384([
                    0xab4378c307911523,
                    0x57cf38b552d24419,
                    0x8e9e83d8ed743aaf,
                    0xc84bc1e5ff888b16,
                    0x07f29a19bddcc0ed,
                    0x10d0bd91d0577974,
                ])),
                Felt::new(BigInteger384([
                    0x45f0a89fd3832053,
                    0x3ffe286e94eaf042,
                    0x1df16f0185dafc72,
                    0x2530fac9c337d433,
                    0xc5b6df9cd03af86c,
                    0x0d5792fd882712f5,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x0dcdef22c1ffa1a0,
                0x3ac35966ecf6acb3,
                0x4f7617c809e278f8,
                0x9688caf280e71276,
                0x0a19ca1ce99d0303,
                0x18ba7007a817c608,
            ]))],
            [Felt::new(BigInteger384([
                0x0592f51731e00920,
                0x80a55b6a265144f1,
                0xaf4c1dfe33229bef,
                0x1db11ac835be7eaf,
                0x400758a882253c76,
                0x0b3021d6b89481e5,
            ]))],
            [Felt::new(BigInteger384([
                0xe3c81e62b7013208,
                0x0598182b1254c8d3,
                0xc6da5a90aae5621a,
                0xbce310efed28ca9d,
                0x025250480041102d,
                0x14cd6020f4ed3e4e,
            ]))],
            [Felt::new(BigInteger384([
                0x6436c235d553a6c5,
                0xe8235e84a319f0d4,
                0x84988a8d22c91f04,
                0xae1e25e5620acfc0,
                0x09c529610c20f12e,
                0x00982c092afb37b1,
            ]))],
            [Felt::new(BigInteger384([
                0xbda4c6873e235a68,
                0x26e0fef61f3a2b85,
                0xdc9f5d2000316425,
                0xab37054b112381b7,
                0x79327acbe8c36f5c,
                0x043f01243cc47c8c,
            ]))],
            [Felt::new(BigInteger384([
                0x4eeee19500baeaf7,
                0xd2470bae6843410a,
                0xd9faa1d5b19c0e7d,
                0x4ad81f820616ed66,
                0x9d1a9dbc8021be23,
                0x0279efa3bf9448d9,
            ]))],
            [Felt::new(BigInteger384([
                0x8fc2326b1f28c9ff,
                0x249899f17f63c8ef,
                0xa5b24151dbbd66c1,
                0x98105827bf15303e,
                0xdbafb0870eb2607a,
                0x19abc81268abba33,
            ]))],
            [Felt::new(BigInteger384([
                0x474ca39e5c181b4b,
                0x5e3bad6a34c14789,
                0xbb95c250ffda29e8,
                0xb23e7c4426c19cda,
                0xd1aada5e5f73769f,
                0x06c773433c12e362,
            ]))],
            [Felt::new(BigInteger384([
                0x31d47cfc8f3f4e36,
                0x67d042bf979ee216,
                0xe7632b54de55d24c,
                0xbe1458a9faa30ce8,
                0x5363644fb76d3d18,
                0x175224fdb17fd403,
            ]))],
            [Felt::new(BigInteger384([
                0x9784c79538db8a27,
                0xbdb853dd2bc0a716,
                0x8798a5b8f4558f27,
                0xc01cd0baa695fcd9,
                0x0664ca4b2a43ead8,
                0x08f775d366749202,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
