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
        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
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

        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
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
        // We can output as few as 1 element while
        // maintaining the targeted security level.
        assert!(k <= STATE_WIDTH);

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
    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger384([
                0x03a82bd7bfe65512,
                0x04f5b5d096a93dd3,
                0x8b0ab910480f9b3b,
                0x02357a0157d397b4,
                0xc9866581d182052d,
                0x0064e4b1e1994fe6,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x13ee21193f5154a5,
                    0xf6e605f260e33ab4,
                    0x832c8530284c56e9,
                    0x3c9268d98cee500b,
                    0xcecfa85755aea575,
                    0x00aca495821c78a5,
                ])),
                Felt::new(BigInteger384([
                    0xfdc94684c539fc75,
                    0x248bb9901d57b995,
                    0x007a28758a2c5e6f,
                    0x22594813c7d0556b,
                    0x9ec5266539ae4349,
                    0x00b5a89cdd535c22,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xc987d49aec495ce2,
                    0x0fe01f72d061bec1,
                    0xd5505430ae1a7fc3,
                    0x7d882be7ac8a695c,
                    0x941e5523f6098a3b,
                    0x0084253496d167df,
                ])),
                Felt::new(BigInteger384([
                    0x385399bb177b25e0,
                    0x66ac5e7e968b52ac,
                    0x63035c34d6b369bd,
                    0x1b1a011e98e35b05,
                    0xe2c4334849700211,
                    0x0122fc8be4ea18c0,
                ])),
                Felt::new(BigInteger384([
                    0x9b38a0776976e744,
                    0x873dcbddc95ca3cb,
                    0xd992bde67218bb9e,
                    0x19eae6b60f9ba3b9,
                    0x0ac8e4bee8a74579,
                    0x00077aeb6fd677f8,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xab0d398e051b8c80,
                    0x866bad3b5062e4b9,
                    0x1ae63104001906aa,
                    0x0fafd51c1a8c0a1d,
                    0x581514cb80398713,
                    0x0025d1d4a2e2401b,
                ])),
                Felt::new(BigInteger384([
                    0x7a5aa3067518cc90,
                    0xbd7de88fa83c7545,
                    0x780bdefedc8ac714,
                    0xa53d8c5a79debe26,
                    0xedc51040aea6484e,
                    0x01556a6c591bf86f,
                ])),
                Felt::new(BigInteger384([
                    0x87be2e19862fd3cd,
                    0x47f0174f93ac2ac0,
                    0xb876e242027c3ae1,
                    0x0977bcf7294334f4,
                    0x219cb9626a984dcf,
                    0x01965c6ef9eb78a5,
                ])),
                Felt::new(BigInteger384([
                    0x2b71aeeac8da4eaf,
                    0x08a696e0bf50c654,
                    0xb717bd90692c5d3c,
                    0x1c317e063a1d4601,
                    0xf585e99365584d45,
                    0x00522ce46950228d,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xd6eb06c9914b2610,
                    0x66c92a6963a58716,
                    0x5a1d62d601a003a7,
                    0x322d48af541f4103,
                    0x8b47bd5c0b545d0b,
                    0x01803e3d3edf7fd4,
                ])),
                Felt::new(BigInteger384([
                    0x9df14f98ec9e8a77,
                    0x6dda719fee931bcd,
                    0x877feed6bb333b51,
                    0x30c75936554b8e5f,
                    0x76242ba621ac45e5,
                    0x00a28d3fbb286810,
                ])),
                Felt::new(BigInteger384([
                    0x73f8b19377100705,
                    0xdae46854963e089c,
                    0x1ed32b146dc947e8,
                    0x50f58cc718a6c198,
                    0x1f61ff740ba88cde,
                    0x01052e7b89dc35ca,
                ])),
                Felt::new(BigInteger384([
                    0xbbc86b7745be11e9,
                    0x7ebce5547dcf80d9,
                    0x66eb34a0536f10fc,
                    0xb6affc8ed7a7b593,
                    0x424bf12fd624bc7e,
                    0x00feabd9ce8e2444,
                ])),
                Felt::new(BigInteger384([
                    0x9df99d9ae78ee67b,
                    0x52f1fe1f97de3768,
                    0x5f3955435ff8a732,
                    0xe04a514ac5d74b86,
                    0x3072b144416784f2,
                    0x0077a340fba53d7f,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xf107766c8e1408de,
                    0x5aa65675851cf95a,
                    0xf55cc7ab278010df,
                    0x4cf5f54381183acd,
                    0x0b1a7288779d66e3,
                    0x0136c89eca24cc5c,
                ])),
                Felt::new(BigInteger384([
                    0xd5e7eec931a78070,
                    0xdf59b9516e4c07b0,
                    0x5b4e5ca0acf09d47,
                    0x8ead2cc37dfcbc6e,
                    0xd13ea55f13ac901d,
                    0x01198274d1c250d4,
                ])),
                Felt::new(BigInteger384([
                    0x0bfa4301b2410371,
                    0xe964dce0a458ecf6,
                    0x9736b5289044ac21,
                    0x97807b90b04f7f43,
                    0xae3e29cce7578920,
                    0x009c17b12af248a2,
                ])),
                Felt::new(BigInteger384([
                    0xe527ba21b29a4c64,
                    0xc97c8d41c8e09aef,
                    0xcea39731a982a7ee,
                    0x2756373853852e45,
                    0x8bcf27fca68c32d7,
                    0x0154dc24e34cced0,
                ])),
                Felt::new(BigInteger384([
                    0x4b6b78d2808a6b93,
                    0xffb67511a4f46874,
                    0xc0d5028c70a3629a,
                    0x3a7cbb82c4077e8f,
                    0xfcae6e1ac1663b97,
                    0x00f7180cf8479542,
                ])),
                Felt::new(BigInteger384([
                    0x9b4b8a3d684df4e2,
                    0x8bbcb137ffda0076,
                    0x656ca3fd5743f1aa,
                    0x56b6f742a2f3aaf1,
                    0x4271831cfb40a6c4,
                    0x00cb32d0d15527b3,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x26bc3487fc9b67c5,
                0x6d3c842076aaa05a,
                0x5987ce024c894a0c,
                0xa1261433b4c0b022,
                0xa3de2988838fd5ec,
                0x0152d286b1151cb7,
            ]))],
            [Felt::new(BigInteger384([
                0x308af42dd03930b5,
                0x6c0728a7fc944112,
                0x368c9491cfaa1339,
                0x95fd39750c1375ce,
                0xbb7572f00b6d7e97,
                0x007c386a047c833f,
            ]))],
            [Felt::new(BigInteger384([
                0x2b093010b934b6bb,
                0x31edad8c5aa92816,
                0xfd59c5239e43040f,
                0x496e6098872bfb50,
                0xa716f5b840eae115,
                0x0160dc97ee84a0b8,
            ]))],
            [Felt::new(BigInteger384([
                0x6a1d670606c27608,
                0xc10413dc5dc43103,
                0xd411bd68273f727d,
                0x3064b77e0a4efd9f,
                0x9a35d6abbc7a4ffa,
                0x00247ef73fcfb913,
            ]))],
            [Felt::new(BigInteger384([
                0xb93881f2d8df4f66,
                0x1f65a14a39e44869,
                0x35bcb9eef4eb8368,
                0xe23c8a9374e17aa5,
                0x1bbbf664e9c1018f,
                0x017595fa260caeca,
            ]))],
            [Felt::new(BigInteger384([
                0x672808acf8740915,
                0x07e8c7fe1af0f604,
                0xcda8973acd59f9a4,
                0x7dfa2f0bb0bf0fb6,
                0x5682fbca35489265,
                0x00ece881c478fa61,
            ]))],
            [Felt::new(BigInteger384([
                0xc30f3264ccc18b41,
                0x8aa25aa004d38fdc,
                0x66b944a822bc43ea,
                0xfe0f582ae6552ab6,
                0x515634c3d9580de1,
                0x0164524355ab8ab5,
            ]))],
            [Felt::new(BigInteger384([
                0x40940ff1b6d3d53e,
                0xf6e15909aeb74661,
                0x7a60a3432cf03b1e,
                0xfe472176e783d11b,
                0x50c7daff05c4d7ec,
                0x00a645591259117c,
            ]))],
            [Felt::new(BigInteger384([
                0x695ff9a74fdb57b0,
                0x2b8d559123cf12ba,
                0x42b241014aa2e7e3,
                0xe07c83bab39a5766,
                0xb862e1c104edece3,
                0x00fae2bab926d02e,
            ]))],
            [Felt::new(BigInteger384([
                0xe76505856510c554,
                0x0311dec230307f84,
                0xdfa4ea20ccc271c9,
                0xdcc79a263044fd49,
                0xdfe99595cba80966,
                0x00c4e7034e09ec37,
            ]))],
        ];
        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa1fbb9b1274a0213,
                    0x637624ac78b468b0,
                    0xbfe24ccece8527aa,
                    0x341512fc6aa42867,
                    0x90a87db385152c1b,
                    0x00b771a3ba82e44f,
                ])),
                Felt::new(BigInteger384([
                    0xd6dc1a08a5e7608b,
                    0x7d0d1b24a3879e17,
                    0x2b5f0d7ac10493f2,
                    0x67763c79f51eca14,
                    0xd6c54a7e2098ce64,
                    0x000f4635ad0ce164,
                ])),
                Felt::new(BigInteger384([
                    0x3983a93bfe756911,
                    0x74f7b8f449cf0750,
                    0x9a2283bb16f32512,
                    0xbc8336f54a845472,
                    0xb2d335a38b055722,
                    0x0097df1c94c6e262,
                ])),
                Felt::new(BigInteger384([
                    0x5ba9fb87ab01e38e,
                    0x428050b08ab24b87,
                    0x7501eff81bf2e39b,
                    0xbbd175caa52d86c2,
                    0xd1841ec8ae589e47,
                    0x012a408a65847e13,
                ])),
                Felt::new(BigInteger384([
                    0x74dccb9ca722cb05,
                    0x40130b26fd199d81,
                    0x740ad68686024425,
                    0x4df342f21b4b261a,
                    0xb5d1c400bf29636e,
                    0x00af99086caa20d0,
                ])),
                Felt::new(BigInteger384([
                    0xdf934893606ab354,
                    0x32985972948c840a,
                    0xa07d5348a0302994,
                    0x161c0f93a6b910ed,
                    0x8658e14eaaee1d00,
                    0x0181d75b13f6eba2,
                ])),
                Felt::new(BigInteger384([
                    0x30bf08877b77218c,
                    0x36aa19f12b66c50d,
                    0x42307bfeca38adf3,
                    0x3d62d3f51fe7e99b,
                    0x16cbd405a458c683,
                    0x00a9eba9f90399a7,
                ])),
                Felt::new(BigInteger384([
                    0xae30db7b14a6ba11,
                    0xc51167a73c6c6356,
                    0x4bda126de3d306e6,
                    0x9f7c75e5c33629eb,
                    0x1ecff29b0877f318,
                    0x002d7099c4e37fdf,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x6ecc12c8e870f6d8,
                    0xfe574dc44b98ebc6,
                    0x66404aa1b67389bc,
                    0x64ed1f94958ef16b,
                    0x09b9e84bdcdd54cf,
                    0x01923d1c133cae40,
                ])),
                Felt::new(BigInteger384([
                    0xced909e49185dbc0,
                    0xcb1a3afa0b8dfa50,
                    0x048ccd3c2a07b049,
                    0x0cfe1aef7c4772ed,
                    0x96b12038bba404ce,
                    0x00765bd6530cb181,
                ])),
                Felt::new(BigInteger384([
                    0x260b0f2b794201de,
                    0x2dc4e11cb542d8d3,
                    0x5c84c569bcf0f162,
                    0xf9ae671fe2b7bbd6,
                    0x0c549cc6b0718c50,
                    0x018bab0f9fca1ebf,
                ])),
                Felt::new(BigInteger384([
                    0x84d7cf24da922b16,
                    0x2935722ae42085fd,
                    0xdebdd9ef10a5a634,
                    0x14026fea0c962e40,
                    0x850bb510ae50af30,
                    0x010da308c07fdb29,
                ])),
                Felt::new(BigInteger384([
                    0x314761a5e0235d3c,
                    0x9d72063345ab662f,
                    0xd8ad41465fb1544c,
                    0x8f4caf8dd4068b11,
                    0xd4d6a1fcfeeb73ab,
                    0x00fb7d30771ec1de,
                ])),
                Felt::new(BigInteger384([
                    0x74def13ea8386f4c,
                    0xd88044ff79fe41b0,
                    0x99ab1ea4e14541e0,
                    0x99a499b8b5d88476,
                    0x512c70e6dd857a83,
                    0x01858078d22ad621,
                ])),
                Felt::new(BigInteger384([
                    0xeb081adf203f041b,
                    0x01ea2679a88ba148,
                    0x4d32f5389bcfee58,
                    0x9795694c3162753d,
                    0x2799261bde2e4a46,
                    0x000404eb91fffec0,
                ])),
                Felt::new(BigInteger384([
                    0x818b7170021e0c48,
                    0xb0bdbff7554f8a6c,
                    0x97c08f843cb19b29,
                    0x34225ea44a0f171c,
                    0x5205f971ff006da6,
                    0x014423728abd456f,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xaabd1a6f5d4b1965,
                    0x5288ab1ac4991693,
                    0x017fe4e4ccdc42e7,
                    0xfa279459bce02874,
                    0x127bf1410e12beb6,
                    0x014161d790dd6d6e,
                ])),
                Felt::new(BigInteger384([
                    0x747682eb1601ba84,
                    0xdc0164d4a2bd5d37,
                    0xf87c905be55a4885,
                    0x16cc30ab000848ad,
                    0x442c00ee29b0b161,
                    0x004fc8f707939be0,
                ])),
                Felt::new(BigInteger384([
                    0x49f210d278ac58ca,
                    0x6d5766deb5eec69a,
                    0x692a06a0f995dab9,
                    0x20440ead8d4863f0,
                    0xbbd61bf1b736e34e,
                    0x0031d38915518923,
                ])),
                Felt::new(BigInteger384([
                    0xed2ae6bf53a96a66,
                    0x36f040a7ccb9ccbc,
                    0x004ce30d11a4c4eb,
                    0x05df04f2892713e7,
                    0x5d3cce734ecff547,
                    0x018af73316fcdca9,
                ])),
                Felt::new(BigInteger384([
                    0x27db5408de01f6a1,
                    0x742fd51155d5a988,
                    0xa487bb2a0a6972ff,
                    0x500ccbde62ede756,
                    0x5033229dfbfa9f56,
                    0x00851f93e84e6376,
                ])),
                Felt::new(BigInteger384([
                    0xb20da00000a274a9,
                    0xc0a927d45c81aed9,
                    0x4fec63acae6105f9,
                    0xae183f70fc10c5ba,
                    0x7fa4438fea8c547b,
                    0x00e5974b05f4bad9,
                ])),
                Felt::new(BigInteger384([
                    0x770d66e6b397d2fa,
                    0xfb6d7f0dbf91e95e,
                    0xfa93afb340ca7fad,
                    0x4c79bc3474a8d52f,
                    0xbc497a30d80f887f,
                    0x0101608490a4c558,
                ])),
                Felt::new(BigInteger384([
                    0xb9dc545ee209b602,
                    0x4efad8bb86c2bd43,
                    0xb3fcc1f8648ccce3,
                    0xc01a3fa1d0ebd1fc,
                    0xde688d20fdeaad92,
                    0x009fdbfcfbea42af,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xe3ae78b062bb1a2e,
                    0xa9269b14bec8267f,
                    0xfbae6b5384a3c5b8,
                    0xafa005ec63f37e67,
                    0x877042a7d8deeb36,
                    0x00db560a4241bf8f,
                ])),
                Felt::new(BigInteger384([
                    0xcc1923a8a784080a,
                    0xecea6b83e86bf44c,
                    0x6aec48f15475e64b,
                    0xd528424d89bbabf8,
                    0x3a980dd01d875316,
                    0x01196dcb35ab9e85,
                ])),
                Felt::new(BigInteger384([
                    0x9f4bc65a48c9e018,
                    0x3fb9e93f9a26e0b6,
                    0x5214fd3a3dc8b6c1,
                    0xb18a2826f61b8605,
                    0xbf11560ce05e6fd7,
                    0x0024e593c89e6a11,
                ])),
                Felt::new(BigInteger384([
                    0x366dd1420fdefe67,
                    0x0afed80553b72356,
                    0x9232efcc25c8a6b4,
                    0xa30e1e84e02ea663,
                    0x17e13664371b7024,
                    0x0054b822a42b4409,
                ])),
                Felt::new(BigInteger384([
                    0x0cec9196c2cb8fbd,
                    0xe7849c64018ab44c,
                    0x269a205d13fd7179,
                    0x4c67bd102e86ef06,
                    0x6e82ccdfff7fac6a,
                    0x0086e3e4aae13197,
                ])),
                Felt::new(BigInteger384([
                    0xd6e21944d473b5b5,
                    0xccfba13c645a01cb,
                    0x6a140648fc88938d,
                    0x8862dcde0088a8dc,
                    0x5b8dc9ac1c394004,
                    0x01a0ee57635f1197,
                ])),
                Felt::new(BigInteger384([
                    0xc49082b0e839068f,
                    0x9885f80f11775a57,
                    0x29bdd3c4f9f95b32,
                    0x592eb65803b1e57d,
                    0xc5b01b91513928cb,
                    0x008ab5cc214b0881,
                ])),
                Felt::new(BigInteger384([
                    0x76cc700fe77a2056,
                    0x799d55b2bc66de8e,
                    0x5c09fa64cf0b601d,
                    0xa6674976df42331e,
                    0xfa16b620dd90eac1,
                    0x012a070c73fb3b22,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x737db3eeb8ec978c,
                    0xfd237ef8cbbe1224,
                    0x68189db04da31db4,
                    0x096671dd6ab1861c,
                    0x470caa4f56c540c3,
                    0x001dacf50f4b43ff,
                ])),
                Felt::new(BigInteger384([
                    0x51b3a5397d9c97fd,
                    0x08f7b7c08dfe87a7,
                    0x1489ddcaed17231a,
                    0xa80a1c7a1443ea3d,
                    0x6d3d29a199b78f6a,
                    0x00589fb7ffbab8ca,
                ])),
                Felt::new(BigInteger384([
                    0x74c0fe3e925377cd,
                    0xa0cb54ce36a25134,
                    0x3f39a38357e40cdb,
                    0x5e7db41dbabed960,
                    0x59cd3ad914d8a4e3,
                    0x0063e0784784d40b,
                ])),
                Felt::new(BigInteger384([
                    0x5333cfff25d7d4d7,
                    0x8fab86e69ac454b6,
                    0x2cbc32bbaefbbdd0,
                    0x63fb164cca2a76d0,
                    0x3ccfffacd41b2184,
                    0x00c643c73aa741cd,
                ])),
                Felt::new(BigInteger384([
                    0xe2b51df094033f6e,
                    0x0b07058c82655c66,
                    0xac02c500762d9af5,
                    0xe34dc4f0af3a31b9,
                    0xe756a7f0eba73c53,
                    0x0088dc01465957b7,
                ])),
                Felt::new(BigInteger384([
                    0xf8041c42dd0e1468,
                    0xa87dad4af9010cd4,
                    0x376cef5c6e88be8b,
                    0x7137a654292994ae,
                    0xaa7fca8dafdb6687,
                    0x0132eae25a6d7455,
                ])),
                Felt::new(BigInteger384([
                    0x20029c668024b878,
                    0xf36ec36d652625ba,
                    0xe87c457a89155c4d,
                    0x0e63d6b143556818,
                    0x38075148e5b22bdf,
                    0x0184da600cd7e98b,
                ])),
                Felt::new(BigInteger384([
                    0x72cca02368c41a00,
                    0x9b2bcc411ae3b2b0,
                    0x3718fcca5be5c69d,
                    0x13133369bee13e3c,
                    0x3198e88c82631c14,
                    0x00df994f65a5cd6c,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x0d98d6171ffbaab8,
                    0x4b001bbc20976b89,
                    0x479e180aaaa26b73,
                    0x5d960e5cf6c862b8,
                    0xe15ec7829e0d26de,
                    0x013f988a8c5ffcc5,
                ])),
                Felt::new(BigInteger384([
                    0x6ed0cab81494d334,
                    0xb03b979443167e8b,
                    0x49a38ef2bd4a8b79,
                    0x52d548f92d9620d8,
                    0xedffb14348c42e26,
                    0x013c6dea15d0e388,
                ])),
                Felt::new(BigInteger384([
                    0xf4cc24debea7fed2,
                    0x97f85d84daa40548,
                    0xacd5707b83f79e8b,
                    0xcc23b526f19ae293,
                    0x6d6fcaa181f657da,
                    0x0148a2701e87c0bf,
                ])),
                Felt::new(BigInteger384([
                    0x6f281dfd3ca27b4b,
                    0xa56af9ac78ac6ad0,
                    0xcca55d7655e77074,
                    0xe2954612644597ae,
                    0xae640199e11bee5e,
                    0x006100407a4f5233,
                ])),
                Felt::new(BigInteger384([
                    0xe922154e50de7f54,
                    0x7cdc3b6f4966d151,
                    0xcf47c9ceb4e039d9,
                    0xd21a67202c3737fe,
                    0x0d67bf09d8ae55b1,
                    0x01adbb0c4b91865a,
                ])),
                Felt::new(BigInteger384([
                    0xbd9f28e1b4cf9228,
                    0xd545452160223965,
                    0x42452108f23ed749,
                    0xfc5605d2ac96a795,
                    0x1b2e64f7670fe725,
                    0x0086aa6fe2b8ce46,
                ])),
                Felt::new(BigInteger384([
                    0xb5c8f19a3484a1a2,
                    0x7559e690fa6facfb,
                    0xada215238f97d730,
                    0x068331a8e419d09a,
                    0x69614fddbfd64196,
                    0x01978646c20c8efa,
                ])),
                Felt::new(BigInteger384([
                    0x8c5779b95119507d,
                    0x5d1abb9e6b10a22f,
                    0x298f3ef9a373cbd2,
                    0x82dc7affc6c977a8,
                    0x326d09a50a91a961,
                    0x007931ae930cc2c4,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0xfdac690249136593,
                    0xd0bd369904c66477,
                    0x8659fdd3ee6de55c,
                    0x14ad53e0f67f5dd4,
                    0xa86441e4532e6d3c,
                    0x00fa6fdab517c13f,
                ])),
                Felt::new(BigInteger384([
                    0x319ebf5a00373d28,
                    0xe01dc5b84731a8ee,
                    0xe4e378958ecaabee,
                    0xc56b7c7b09c93dd3,
                    0x79f5149bafca3449,
                    0x00330b33bfe32924,
                ])),
                Felt::new(BigInteger384([
                    0xec01dc2485355f80,
                    0x1509981847f93b2d,
                    0x26263762d7d849d4,
                    0x8da5e5efa844cd95,
                    0x6f0ff624d4d889eb,
                    0x00b387b634ccbf7b,
                ])),
                Felt::new(BigInteger384([
                    0x46cc2b5dc254cf10,
                    0xb82625358699d918,
                    0x694c4a5bd1bd640a,
                    0x64becd5e2c5ccef1,
                    0x9d4fcb94b0eeae70,
                    0x01633bdc22418ed1,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xbee5d927cb325d8d,
                    0x7576ac963f9610f6,
                    0xd4b27edf431adf09,
                    0x2efa6421a6a02f30,
                    0x4495141c98d58bd9,
                    0x00ca7eb7139a7fe1,
                ])),
                Felt::new(BigInteger384([
                    0x994a1f9fa47a0524,
                    0x5e0890fa9b41cae4,
                    0x0f0cc83bc2f272ed,
                    0xac1ef182f1ee20cb,
                    0xbaa6d3ca965bdb9b,
                    0x00fc1893089405ab,
                ])),
                Felt::new(BigInteger384([
                    0xe4c1f013e215dcd0,
                    0xdb407a20046920c8,
                    0x3b9414c660e72cba,
                    0x30a9b07ba70bc6c9,
                    0x3c6bd2684646239a,
                    0x01614debac4e314f,
                ])),
                Felt::new(BigInteger384([
                    0x6fbc408589be5d8a,
                    0xca4cc1c7cff8fee7,
                    0x376504fe45cdab08,
                    0x35b60b14d22a1348,
                    0xfb40e0573a0f6640,
                    0x00681cd37a521596,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x64ba61c8a4ff40b1,
                    0x15d063b15fd3d3cd,
                    0xf018da218064181b,
                    0xea1a192221c857df,
                    0x5c7928ee35d16fa9,
                    0x01854e779df90f9c,
                ])),
                Felt::new(BigInteger384([
                    0x72ce3b610776cc32,
                    0x2139625cead5838f,
                    0x3ecc42be5c6d46b7,
                    0xcf0d4c2abf8d14cb,
                    0xc6202eae48af334c,
                    0x0149f4b0951b0c7a,
                ])),
                Felt::new(BigInteger384([
                    0x1e3d27f631b1bccb,
                    0x7368b2a6b364b902,
                    0x08a8ab8e26906320,
                    0x2581135fdbe38d2d,
                    0x098ef7a0c9dac2d0,
                    0x00177001d3238cc6,
                ])),
                Felt::new(BigInteger384([
                    0x1cf1ce6cefe15e11,
                    0x5471f125a66dc9ac,
                    0x139184e748fc556a,
                    0x25d640f81de83824,
                    0x81c86cd297cfc728,
                    0x00e5e1952bc55164,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0db8f9e63a445e75,
                    0x02b627abb6dfb3b8,
                    0xbf7cc050726c83c8,
                    0x3b07b408d6e827c3,
                    0x72d912f2f3bf0e82,
                    0x013a53e85bc81ce4,
                ])),
                Felt::new(BigInteger384([
                    0xb658ddd5d3801ce2,
                    0x3bb2ca52385613c1,
                    0x07f3a47144959020,
                    0xc7769dd0793f766a,
                    0xc9f1c722a37c98da,
                    0x007f90aff3af0829,
                ])),
                Felt::new(BigInteger384([
                    0x1f290ebf462f4267,
                    0x053fc988be2475ec,
                    0xf89f57f7abfd0eb7,
                    0x5c3a8678cf3614af,
                    0x6735c6d7598cefb4,
                    0x01804120580ab7cb,
                ])),
                Felt::new(BigInteger384([
                    0x25c3f3c6b7ac2f72,
                    0x836e7f86458c3097,
                    0xd3585edeee057b22,
                    0xb40c66f5fd6a144c,
                    0x99a4a6750ff3f4e5,
                    0x003f53e3c6d00c17,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x1c9acced3088c6b6,
                    0x9663782102a6ff55,
                    0x0ed5f8ddf604c19b,
                    0xc6432347e55429c9,
                    0xb965d69364804c92,
                    0x01580e2440bba054,
                ])),
                Felt::new(BigInteger384([
                    0x6280223baed59397,
                    0xc3af8a694e8459a4,
                    0x0b9758850471805e,
                    0x735f6cbe81b5b0b9,
                    0x2e7c0611480fad34,
                    0x010888e9f2f4c203,
                ])),
                Felt::new(BigInteger384([
                    0x8cb39243ff916a11,
                    0x1664bec47c98d224,
                    0x7d0eeae8f0b205f6,
                    0xb70ff6e87074fecb,
                    0x223a7735685d72ee,
                    0x01a1034954594653,
                ])),
                Felt::new(BigInteger384([
                    0x3f370ed35d152b83,
                    0xa837e1f7c0c0dc47,
                    0x2ec2afa8457b1202,
                    0xce69c9fe86b65329,
                    0x26bface1d710b747,
                    0x002b830f82607721,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x27557df77808a7be,
                    0x4c520cc240144179,
                    0xcc04ff4c150a9867,
                    0xfdcc033f257051a5,
                    0xfe7a86dac7cb7bef,
                    0x00dbb14406590361,
                ])),
                Felt::new(BigInteger384([
                    0xb08a52b75bf5feb6,
                    0xd977756b896c74e9,
                    0xf6fa584c36fbf814,
                    0x7efccba87d64560d,
                    0x38d186892470dbf4,
                    0x0030658c41d67efc,
                ])),
                Felt::new(BigInteger384([
                    0x7401a39a5eb88891,
                    0x9d7132f18b9eff30,
                    0xb17be2794373aa0e,
                    0x71e518e458532df7,
                    0x1dee8c34196b98f9,
                    0x00d9afd9e8255118,
                ])),
                Felt::new(BigInteger384([
                    0x71655376280b0ca3,
                    0x10415bafe4ed627e,
                    0xa75890515323ff25,
                    0xc93b51ef8729b494,
                    0xa2ee301dd45e2815,
                    0x000d67ed3b17220b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xbe98e5bb4b5f6dc2,
                    0x3b40b414fdc516c4,
                    0x2f564d548e2ca3ae,
                    0x2cd12a5766bc76b8,
                    0xe6135e1ea0167432,
                    0x0172d3f8a4690ed9,
                ])),
                Felt::new(BigInteger384([
                    0x23c30d6d79ec775e,
                    0x8f19c93f91d7bed8,
                    0xadd8a2a22be0625d,
                    0x38df8f01caacf0c5,
                    0xc6dfdec789dde55e,
                    0x0116a148c3889cbb,
                ])),
                Felt::new(BigInteger384([
                    0xc79a96632cc494a5,
                    0x556184fcccf7c445,
                    0xb04e4c783d8f17b5,
                    0xc94dc59fefec6792,
                    0xec04b3ca9d423e9a,
                    0x01740cd4d0789673,
                ])),
                Felt::new(BigInteger384([
                    0x3cf3e15a913b0d1d,
                    0xdb0f812f1d81ce70,
                    0x91ad4407ee1eea09,
                    0x4ba365417a2134d0,
                    0x713daed36761f7e3,
                    0x01154651d39d1121,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x005c30cf344c33f2,
                    0xaa296c57d2875bdd,
                    0x708155d452107ba0,
                    0xa6554fdae3e3060c,
                    0x49e4831d9300ae11,
                    0x009d6ce71bff1008,
                ])),
                Felt::new(BigInteger384([
                    0xd19f286c54cb9264,
                    0xdec0b9d0bb2767e6,
                    0x4064ee428a30a65f,
                    0xc01ab97592d4b960,
                    0x7ecc98679b94f88d,
                    0x0033e61c4d3a5d4f,
                ])),
                Felt::new(BigInteger384([
                    0x87015e99e77d6fbe,
                    0xef967382c5f900e0,
                    0x4889d495baf64067,
                    0xfc387f9adac70608,
                    0xb5bf0c8e408292a1,
                    0x0020e4d63f2cb372,
                ])),
                Felt::new(BigInteger384([
                    0xddef6f4ed5b6d70a,
                    0xe8c8ffc6eb19e1ef,
                    0xfcc1b9dfefe9fe6a,
                    0x8757a6f18d322535,
                    0x5cf61954393ebab9,
                    0x00fd44d0343cc123,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc3541a7a0f8db46a,
                    0x522592a12198be46,
                    0x1fb9b98e6fd89efa,
                    0x42d6080e3afd2d15,
                    0xdb6f3a9c294c15bf,
                    0x00eb54fd0924ffb4,
                ])),
                Felt::new(BigInteger384([
                    0xb76aaae298ea0e30,
                    0x8a266d3a262b322f,
                    0x332cfdb6c79de541,
                    0x7f517a0b615d023e,
                    0x2aaa43ece3c99133,
                    0x012889dbed9a654a,
                ])),
                Felt::new(BigInteger384([
                    0x63baeb1968ab6f6c,
                    0x5b7421f4bb20af03,
                    0x88b1a4969e7bc3fa,
                    0x7a8e2a421186908c,
                    0xad3591042dcc83ec,
                    0x00671fcab25e8e83,
                ])),
                Felt::new(BigInteger384([
                    0xfee6cbd3f3a6006a,
                    0x8c95e553ccc18277,
                    0xc31d98f781e01271,
                    0x87b0f5cfa941264b,
                    0xd31d62252ea44c16,
                    0x00ba1a938c2d94a8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xba45f371582ca06f,
                    0xb8a0d4c7a22175a2,
                    0xb069cc82a65800de,
                    0x3f6ea0dc3381e56d,
                    0x0c3bad574cd2b983,
                    0x0013259d546beba2,
                ])),
                Felt::new(BigInteger384([
                    0xe0b9d11bad275b88,
                    0xa1fbbc5a242e4bab,
                    0xde0b210092a6467c,
                    0xfeb46cec55ab6ca1,
                    0xdf8324cb6097acbd,
                    0x00cc719d4dc66024,
                ])),
                Felt::new(BigInteger384([
                    0x9b9d440d5a9c5fcb,
                    0x509d6106a219fd13,
                    0xcf58f11aa27dcd7c,
                    0x003ad1c94791a9b6,
                    0x92c5bc7af9ce5ed3,
                    0x009f1282d5bf7330,
                ])),
                Felt::new(BigInteger384([
                    0x116543e9f41c09b2,
                    0x45a4abc3757c752e,
                    0xd5ff4d62e59046d1,
                    0xe9ca6eace8104c89,
                    0x7a7088233f4884eb,
                    0x00e8e856ebf3a746,
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
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x44c40828776aa7bc,
                    0x54d387646035921c,
                    0x8e2d89eb085c1d53,
                    0xe5bd36ea5338aafc,
                    0x403f948a793e8eb4,
                    0x0081a60c87ffa2de,
                ])),
                Felt::new(BigInteger384([
                    0xb1a2397d8f940df8,
                    0x1628251a4432180d,
                    0x34e86be562bbee94,
                    0xba30486435717c22,
                    0xd85ddfd3ebacf1b6,
                    0x004bf420425b1783,
                ])),
                Felt::new(BigInteger384([
                    0xd0b1e3939a987f3c,
                    0xa6deaf8240bf66c8,
                    0xc613742b79a60983,
                    0x89bf2c0aba34d5b5,
                    0x1650a5ff04e98f59,
                    0x00bbb030ed1463d0,
                ])),
                Felt::new(BigInteger384([
                    0x2b71f0c4321d4791,
                    0x8483e40bb893defe,
                    0x53c938b57ce14f95,
                    0xcb2c7a2474ebda97,
                    0xdd435919282e3c06,
                    0x000ce48cc1227317,
                ])),
                Felt::new(BigInteger384([
                    0xbe42fa639f7a7b1d,
                    0xafc8213f144d9a93,
                    0xa2691e6dd6c647d2,
                    0x9f3536e1c1de8df9,
                    0xdd8c3e77a59dfdd7,
                    0x00e8bafd3d448fed,
                ])),
                Felt::new(BigInteger384([
                    0xfab892a217088b69,
                    0x4d9f613e00775c35,
                    0xb884d970223ad23f,
                    0xcd298a3d2008bb0a,
                    0x537d0114f54d57d6,
                    0x01a4ea3bfc326be9,
                ])),
                Felt::new(BigInteger384([
                    0x0de61b461953cd16,
                    0xa586f6180cf7647a,
                    0x1a9a38b946646af7,
                    0x2f28040839515160,
                    0x71a61d6516b2de9b,
                    0x00ac23166176f27c,
                ])),
                Felt::new(BigInteger384([
                    0xb011c6150c2aa6e4,
                    0x3651a07759915404,
                    0x86ab9ac79fecc8df,
                    0xd9f077f646b093f3,
                    0xd46caf64f92ecfd5,
                    0x01353a92046a3d30,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x9f2e29c986d721c4,
                    0x4d83a98293272ce7,
                    0x778a04be0a0edff6,
                    0x09e6eeb7e4c93ccc,
                    0xa923b1eb94fd972a,
                    0x00ff4c1db67b9382,
                ])),
                Felt::new(BigInteger384([
                    0x455ad935350ca391,
                    0x425cd68a8792f8a0,
                    0xd169e90003496018,
                    0xeb4de15de41b3f54,
                    0x8a6f48071d3b1d7e,
                    0x004e89d06ef41ac9,
                ])),
                Felt::new(BigInteger384([
                    0xb4cba1126473761c,
                    0xaca2273d3cbf3bf2,
                    0x43dc10f0cc9f02c5,
                    0x9a303f6eae926a19,
                    0x9b0c9f61436ba57f,
                    0x010fc963315dec6a,
                ])),
                Felt::new(BigInteger384([
                    0x92e90b8cfe19c28c,
                    0x8ebbd126acac957b,
                    0x3ead48119f447381,
                    0xe1ec1d892586bf1b,
                    0xb93583776a649a04,
                    0x00a538d123dd830c,
                ])),
                Felt::new(BigInteger384([
                    0x8ae1eddd5d502224,
                    0x9b138260b372487a,
                    0x1935020c56370964,
                    0x52d4febe11314d03,
                    0xc99a0ac037df4266,
                    0x00e4166f3291d8fd,
                ])),
                Felt::new(BigInteger384([
                    0x1a1dd284a4e7201e,
                    0xd654c3a8f8e7123d,
                    0x74bf8fd061f37cd5,
                    0xdf7608ff31ded853,
                    0x87449537d845019e,
                    0x00b81a3f25e14c79,
                ])),
                Felt::new(BigInteger384([
                    0x0fc234a674bef978,
                    0x74f4d7908aea1f10,
                    0x6b86dce8c1ee2cf1,
                    0x41c9c72d73fb8cc0,
                    0x7f74bd4e1146f52a,
                    0x0170f3f4530c8012,
                ])),
                Felt::new(BigInteger384([
                    0x68df7135c3969470,
                    0x182f7764e1b70c41,
                    0x9edeace6b4c44b38,
                    0x524b9fff5a6742c8,
                    0xf7858cc19344448d,
                    0x00c3ef497234f1b8,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xd3039d53d8778818,
                    0xd312a50e5bf90db8,
                    0xb4963edfb4671111,
                    0x67e57a038b4ab43d,
                    0x063c026ba9a3d71c,
                    0x003388588f5b8fee,
                ])),
                Felt::new(BigInteger384([
                    0x66e41b1e2055b00f,
                    0xf10819b73a1f0a4a,
                    0xf5cd49c2939e35ee,
                    0x9267f85843679c97,
                    0xc0b86150d49cdefe,
                    0x009f565fbac1920a,
                ])),
                Felt::new(BigInteger384([
                    0x2c39d9f1f77e27c5,
                    0x8a842aea21c558cc,
                    0x298ce4a37e742128,
                    0xe77d4edf57987bfe,
                    0x8700316356e4cb62,
                    0x005b241b755b7dd3,
                ])),
                Felt::new(BigInteger384([
                    0xbad0a31d29b0c084,
                    0x9dc2672827b4bb4e,
                    0xf14c47176ed4dfd9,
                    0xab15c15fa751f3be,
                    0x6e35827f4b3a7c33,
                    0x01645d8b93bca871,
                ])),
                Felt::new(BigInteger384([
                    0xe32d254f9941edc3,
                    0xec3ac1e1729640fc,
                    0x51810b93c4a03e1b,
                    0x11d319dbd50e6327,
                    0x25cba7e4f115a859,
                    0x003ddb42fde5aa8e,
                ])),
                Felt::new(BigInteger384([
                    0xe41624c7b1fdab72,
                    0xd4dcceea9e514207,
                    0xd738a7454bfeefec,
                    0x6158144c9274f9f6,
                    0xa8d60b0bf21e1f15,
                    0x017cc48f618239bf,
                ])),
                Felt::new(BigInteger384([
                    0xc2a8f5942dfd636d,
                    0x0b93a7fa7fdfc896,
                    0x6e88d52eb7a3a315,
                    0x6e7a581dd3c6b35e,
                    0xa741402ba45545e5,
                    0x0010206407175221,
                ])),
                Felt::new(BigInteger384([
                    0xcd1d28007671d50f,
                    0xb00cc8356d981032,
                    0x3f000d1a99ba38c5,
                    0xb227e62b0f324822,
                    0xf9c1d091587b594e,
                    0x0086c17f1db8fc61,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x69b7e84670b13691,
                    0x4633e183e99ce4d3,
                    0xc9b0994104ed19bf,
                    0x4878c33f6670dd75,
                    0x44ad8a1adc0f8a87,
                    0x009a14ae7a2663c2,
                ])),
                Felt::new(BigInteger384([
                    0xf28acab4d0249357,
                    0x489557b108fb7233,
                    0xb22c2122a2cdec31,
                    0x2290a843d52443a9,
                    0xdde227fd98008ee0,
                    0x013440738e15703c,
                ])),
                Felt::new(BigInteger384([
                    0x20a90a26e537528e,
                    0x606c74950961077a,
                    0xf0e41f87963bb6ce,
                    0xdfc1e57220cfd353,
                    0x67e995321b7ca761,
                    0x00264de7ec16f3ca,
                ])),
                Felt::new(BigInteger384([
                    0x1fabe8634e4cf12b,
                    0x53ef7d7ee7180bfa,
                    0x2d7fdf26e4c7ba23,
                    0xf267ba5f7fe7f0b2,
                    0x31a0bd099728e4dd,
                    0x00124e27e9f6b2d8,
                ])),
                Felt::new(BigInteger384([
                    0x2b5b6568975a381d,
                    0x72c6a5f1fcf022b4,
                    0x93f76d0ce93329b9,
                    0x783dd11b763d3792,
                    0x6d8c6f2441bde6ad,
                    0x019811ee75251265,
                ])),
                Felt::new(BigInteger384([
                    0xf6a1c6f6f871ce27,
                    0x3bec8508c824efc9,
                    0x34c3b589df3c9aa6,
                    0xd2914480f238d35a,
                    0xd16903838f2f30c7,
                    0x00905d9ad88e7986,
                ])),
                Felt::new(BigInteger384([
                    0xc4e4c81c9449ae11,
                    0x4ba1c617e0a1119b,
                    0x2ed8e0ecb2aa9bef,
                    0x969f605d34038b70,
                    0xe11019417cdceb76,
                    0x008295b06c478afe,
                ])),
                Felt::new(BigInteger384([
                    0x790ad022d02c5e76,
                    0x88a4747063d92679,
                    0x827742a09aa1d8d7,
                    0x8aaf787b5bc75898,
                    0x76123b649756d48f,
                    0x00dba4c98f8cb2dc,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xffd06da43ab90820,
                    0x45056442f0f9a263,
                    0xc2a6fcc6d7c5ae0a,
                    0x4544d3aec0035be8,
                    0xbffc0c1d409a445d,
                    0x018a1ec49d5142b8,
                ])),
                Felt::new(BigInteger384([
                    0xec76a2c810cebbfd,
                    0x811c14cefc6a4d51,
                    0x2ddcfdb94979e047,
                    0x467f17650bec08f9,
                    0xac7ecd7eefba07d7,
                    0x012a7ae307f356c4,
                ])),
                Felt::new(BigInteger384([
                    0x05602e8eaab0cbd8,
                    0x368a32b9098c3918,
                    0x55fc19a64fa2faa7,
                    0xd0d48712e1f3ba0e,
                    0x04683ea6ea6597c6,
                    0x0045d780745bbc1a,
                ])),
                Felt::new(BigInteger384([
                    0x54c1248809d9e5de,
                    0x5bb594043ba4d331,
                    0xb13dc76c6615b33a,
                    0x2212c0bcfdda384c,
                    0xe421f982df5588f3,
                    0x009950ccb94c8687,
                ])),
                Felt::new(BigInteger384([
                    0xf0225ab9f1da32e2,
                    0x2a519907da6e064b,
                    0x271fef63248378fa,
                    0x29bae8400e508c8e,
                    0xc899eb9a068e3f60,
                    0x00dd69b21075e002,
                ])),
                Felt::new(BigInteger384([
                    0x80b274911c58402e,
                    0x32f07cb3badcafba,
                    0xd76fc1aa50e05764,
                    0x2290fb50013d4e51,
                    0x79698311a1035ab2,
                    0x00c1dca091f925ba,
                ])),
                Felt::new(BigInteger384([
                    0xb19dc39ad75a8488,
                    0xbf67d860ba31b62e,
                    0xfc66c82815f6f13f,
                    0x3afb0ed509cb0a8f,
                    0x22b365044f4485b2,
                    0x005eef7a4b93391e,
                ])),
                Felt::new(BigInteger384([
                    0x15dd4482cefeda02,
                    0xfefc175b7c76c5f2,
                    0x9944604020b1cf01,
                    0x9f002871e58f8168,
                    0x9abbffbd089f8f6d,
                    0x00cac6a8345be95a,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x699e899828caf4e8,
                    0xf88c056088fc4cbe,
                    0x3d4578aa31e3eb44,
                    0xac7533a87722c33e,
                    0xf3e6595f5dbc828d,
                    0x00654ee5733f3339,
                ])),
                Felt::new(BigInteger384([
                    0xc283dff1d7525f69,
                    0xfacc54de65fe6ee0,
                    0x162fcd009bb33795,
                    0xd1ef72f997f61251,
                    0x38f0bff240b3592f,
                    0x002b113678ccf007,
                ])),
                Felt::new(BigInteger384([
                    0xa8bbdae604929be3,
                    0x040ded23f27ab70e,
                    0x465fd69356d415d9,
                    0xaba3ac49da29d752,
                    0xb39e94164e751183,
                    0x00acac3c9bc6d5ce,
                ])),
                Felt::new(BigInteger384([
                    0x8ac9d0300b5a4e9e,
                    0xb03bde7493c93e4d,
                    0xc8dda638448534de,
                    0x487ad47876b721da,
                    0x792da4d7a93e8f7c,
                    0x0082f2d77244b591,
                ])),
                Felt::new(BigInteger384([
                    0x7d2a8e3bae344e20,
                    0x92c2d0c36970f6e8,
                    0x7c6c5808072a2a3b,
                    0xdf6442851d04245e,
                    0x33b49bcbbc306a26,
                    0x014e9acf2eb0964f,
                ])),
                Felt::new(BigInteger384([
                    0xda6b4a40b27ed28d,
                    0x79442de3cd357022,
                    0x9f50d75c7a56ec78,
                    0xdb05074b9620b704,
                    0x5400bbcb0e4965e5,
                    0x001a584bd9cce366,
                ])),
                Felt::new(BigInteger384([
                    0x16798d1298e125e2,
                    0x595283f7093ce4ff,
                    0x7b1572b15fcf6fcb,
                    0x2957d24a8c30aa84,
                    0xf233027588b1d622,
                    0x01288dcadfff3a00,
                ])),
                Felt::new(BigInteger384([
                    0x9c3104f374eaef7b,
                    0x9a7572015760b0ad,
                    0x7ed54249ce2cc63c,
                    0xc8a362edf4ccdfc8,
                    0x990d9f93ae5bbb8a,
                    0x0131dcfe7eeabe3a,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xdd106fde90d4d14a,
                0x66ff5c5aea8b21ab,
                0xdbbc95f86cc4f72a,
                0xb25aa9b6d3f5249f,
                0x687e12791c1e90a6,
                0x0196045ab44427c6,
            ]))],
            [Felt::new(BigInteger384([
                0xa29ca960db809d09,
                0x4af5bef04f39fb8a,
                0x18d19c8038af99ba,
                0x0d335d4f0fda02ef,
                0xaa728f25d6445ed9,
                0x00338d7d1344aa9d,
            ]))],
            [Felt::new(BigInteger384([
                0x08a6138cce0927bd,
                0xd0cdaf52447bda0a,
                0x0d3888f5d84b875c,
                0xd03905bed9370ade,
                0x217ab08f06e89a78,
                0x007020330272d86c,
            ]))],
            [Felt::new(BigInteger384([
                0xfeed5a420b9fed2e,
                0x9900808492e66dfb,
                0x55815738dcf20dc1,
                0xde7f8b621adda00c,
                0xb12f3be12779f980,
                0x001d05103ec7c71b,
            ]))],
            [Felt::new(BigInteger384([
                0xff5d009b3f4bef1a,
                0x2130eadb6771f1d6,
                0xbd1f2bceb9dde1e9,
                0xfa01b254bbe5c1b9,
                0x33242c215be17498,
                0x003a438f66ab77f5,
            ]))],
            [Felt::new(BigInteger384([
                0x138ff85499606218,
                0xf5f6db252cf15487,
                0x7500f399cef359bb,
                0x107f263262854aab,
                0x81c4786276db614e,
                0x015e4159f8f8c73d,
            ]))],
            [Felt::new(BigInteger384([
                0xd6ee70a321fbfc6e,
                0xff092a07636337b4,
                0x48feda2b2ebd527e,
                0xc1e391e2e5a1b27f,
                0x916f3d9b37f901e1,
                0x0151795397fbc7a2,
            ]))],
            [Felt::new(BigInteger384([
                0x22941d111919d018,
                0x5e5ec5fa7e0302d9,
                0x8edf283f42c874d7,
                0xdf03ea2c43e4dcee,
                0xe9dea6ba38bfb99b,
                0x000331835d4f1016,
            ]))],
            [Felt::new(BigInteger384([
                0x556405438f515af7,
                0x33e41cc58fc2389c,
                0xef33004320340705,
                0xfaff87038ca637f6,
                0x16743a679f6f417d,
                0x014b7de988991791,
            ]))],
            [Felt::new(BigInteger384([
                0xb075bdfa9fc70c86,
                0xcb6b4d96587c6f2e,
                0xd72a9f3c13081c62,
                0x2767b92ba3743c0f,
                0x269aecd32e43f806,
                0x01033bc3b89d2821,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 8));
        }
    }
}
