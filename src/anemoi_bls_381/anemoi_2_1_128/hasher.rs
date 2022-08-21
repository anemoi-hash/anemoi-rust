//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, STATE_WIDTH};
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

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 48];
        for (i, chunk) in bytes.chunks(47).enumerate() {
            if i < num_elements - 1 {
                buf[0..47].copy_from_slice(chunk);
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
            state[0] += Felt::read(&buf[..]).unwrap();
            apply_permutation(&mut state);
        }
        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        for &element in elems.iter() {
            state[0] += element;
            apply_permutation(&mut state);
        }

        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
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
        // This instantiation only supports Jive-2 compression mode.
        assert!(k == 2);

        Self::compress(elems)
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
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger384([
                0x342e4bb45c31593b,
                0xb541417150a611d4,
                0x2c2ca5129122c114,
                0x7eb0c8a755b9102e,
                0x1e0b47c1cb707135,
                0x0823bfa9b13344ad,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0xc7a91b9ea5efca1b,
                    0xe4953975e90b8a4e,
                    0x1a24749ac9bf1800,
                    0x6f3f7da876fe5e67,
                    0xa28d4959a8957932,
                    0x0f7dcda6742f5635,
                ])),
                Felt::new(BigInteger384([
                    0x262325dd38dcf61e,
                    0xb57227a48da57e6d,
                    0xe30e89a99bea740b,
                    0x8002b39c518a4bcf,
                    0x8a9ab84b50a36b61,
                    0x19762e42ce492adf,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x0a0b687c99e81b3f,
                    0x57f8926eac6f1635,
                    0x531b655d590e9ea7,
                    0xc9d757cca3d6e6d5,
                    0xa284443089b99aba,
                    0x05b2b47c2cbb78ec,
                ])),
                Felt::new(BigInteger384([
                    0x63b453eb2b988daf,
                    0x327941b8509e7b2d,
                    0xd3ca2d4213b57835,
                    0x9ccc15732eb9e819,
                    0xcfeaebb524881921,
                    0x054915a8efe02afc,
                ])),
                Felt::new(BigInteger384([
                    0x6f6a9a4008de4e32,
                    0x121d6ee11e0231cb,
                    0xd6a8840f7a54414e,
                    0xf067e3aa01838d27,
                    0xe84d1c698092fa05,
                    0x15228f165f406870,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x07f58875a0a5812b,
                    0x0f9ee1551d014fe5,
                    0x3718317387097483,
                    0xe8c1aba9fdee2d43,
                    0x34906174a8b8ce80,
                    0x0f5b6c675b11fb80,
                ])),
                Felt::new(BigInteger384([
                    0x26d1bcd6f6b10e46,
                    0x2daafd6341bbd73a,
                    0x384e39870959436c,
                    0x376e610b1bb3358a,
                    0xfc9658378e2e7482,
                    0x0d7b23b4ad693705,
                ])),
                Felt::new(BigInteger384([
                    0xeda93e7ab5090886,
                    0xb9298edebd2aedce,
                    0xc334d64492e57820,
                    0x99eba3040970a1f5,
                    0xe7293032996d80ee,
                    0x064745031e2f3c68,
                ])),
                Felt::new(BigInteger384([
                    0xed09f837e317137e,
                    0x99b4ea3d15d9fdcb,
                    0xcdc3fceac7e2b99e,
                    0x886fec9295348817,
                    0x7e02094a1e673c56,
                    0x126874581992ecf3,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa4cc6f4da4687b3f,
                    0xade90049edb51b90,
                    0x6c0317d4a1681c6d,
                    0xce58366b6dfb155f,
                    0xdc189e3bc2ac76e2,
                    0x187b44b956879c45,
                ])),
                Felt::new(BigInteger384([
                    0xf65c060c49db0055,
                    0x3b55d8dcd2b61c42,
                    0x2f4fd2645cd4e61b,
                    0x56ba4a15a57d33f7,
                    0x127d527e740108ab,
                    0x09d60c4b71b30a1e,
                ])),
                Felt::new(BigInteger384([
                    0xa47e5bec5a4eaa77,
                    0xbad4aa1dd17c813e,
                    0xbbe5ed6121687499,
                    0xd24cc560a28bb515,
                    0x9114c07f8607f12f,
                    0x1357dde3c93261a9,
                ])),
                Felt::new(BigInteger384([
                    0x66324938a4080ea1,
                    0x3b53f982f8ddefb1,
                    0xc77bef321a498add,
                    0x52acc114b2ddb743,
                    0x1fcbd1a946e7fdb4,
                    0x173735a5f33448a0,
                ])),
                Felt::new(BigInteger384([
                    0x95c2283cbd4a8702,
                    0x3c1f933877e05bb8,
                    0xb56a78a115582c26,
                    0xc5476d1ec46fdc92,
                    0x7f12302f22151285,
                    0x07d026348b86ca43,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x66142d67403b882f,
                    0x73f9ffd1c65a4b78,
                    0xe741522665eafdd8,
                    0x1cee43957d01a50c,
                    0xdbf6e00a59ce887c,
                    0x0eb1ab11e74123a1,
                ])),
                Felt::new(BigInteger384([
                    0x3f4c1e185973eb49,
                    0xf440eca607989301,
                    0xf1503df75ed0f1ab,
                    0x30e287304ab4c637,
                    0xa7679132548c30cf,
                    0x03bd0963dc7edd2e,
                ])),
                Felt::new(BigInteger384([
                    0x83fa4b58bcc2929c,
                    0xf8322d31a6c22ea8,
                    0x1cab606001ff08bb,
                    0x1086c4639837e595,
                    0x771fcc8727170cc3,
                    0x0c2fabaa89c38313,
                ])),
                Felt::new(BigInteger384([
                    0x933e27bb968b6d91,
                    0x45646e5ea170d556,
                    0x62a0414e9efbefc6,
                    0xf45ae6424842ba34,
                    0x6e4046cf0c1cee04,
                    0x12cd71299d7455cf,
                ])),
                Felt::new(BigInteger384([
                    0x842a12c309617585,
                    0xcb9d1ca991de914c,
                    0x47b30fd686ff5aaf,
                    0x424a5a7d0e6f403b,
                    0xcb2bff3673d7210f,
                    0x1251dbd85e2163a8,
                ])),
                Felt::new(BigInteger384([
                    0x3de161a672fec288,
                    0xbb6b360f32b4275b,
                    0x64e7715c24ccdf25,
                    0x061489149ade20df,
                    0xde52776db4fdf955,
                    0x1368b2adce6480ba,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xa0332782d923a4ca,
                0x977e38e2977f9f65,
                0x0456d79c9122e4ab,
                0x55035beaa8f786a6,
                0xf11216597a616322,
                0x1701803e9c958c38,
            ]))],
            [Felt::new(BigInteger384([
                0xc777d9cdf28cd310,
                0x5f3b6a002d381551,
                0xbee0b32a9ff4164a,
                0x36e2cd134ff9d831,
                0x380a9ccf8a9e6bb4,
                0x0c68355b260d387a,
            ]))],
            [Felt::new(BigInteger384([
                0x11f1cde0aa8ce4f5,
                0xfab8266e98e9286f,
                0xf4acc64cb2140eef,
                0x726f12b0ce248307,
                0x68a5febd020dee14,
                0x0e580d81d6e882a8,
            ]))],
            [Felt::new(BigInteger384([
                0xbcb2fcfa2a7563aa,
                0xdcadd8386f9f07e8,
                0xc51dd8392e77fc19,
                0x41d284bcaab11b96,
                0x42fd05310d036a6d,
                0x084f1759445d091a,
            ]))],
            [Felt::new(BigInteger384([
                0x4ef2b04425a439b2,
                0xec0c0b8c9c3b209d,
                0x27c9702cd2e5832e,
                0xd5ae3260669294bd,
                0x710567970f421d9e,
                0x080e975996239512,
            ]))],
            [Felt::new(BigInteger384([
                0xa1f099c2d84188fe,
                0x6808523a11479d0c,
                0xab484a24f2cc7181,
                0x43e6c37f39bb3ad2,
                0xac53e092fcbf2b61,
                0x076e4aa96123d065,
            ]))],
            [Felt::new(BigInteger384([
                0xa44bde2e158d6bfd,
                0x1c6bc0409f2fdd3e,
                0xbb912984db9268fc,
                0x51f1e1ceccb308c3,
                0x9302c641c346b3b2,
                0x1487501ade4f8e02,
            ]))],
            [Felt::new(BigInteger384([
                0xa80ce8ddf705cc78,
                0xe43f5b82f4a59867,
                0xa0d7420d43ecfa4a,
                0x939e246c4009031c,
                0xf53e8d54913a111d,
                0x13a43220fc14132e,
            ]))],
            [Felt::new(BigInteger384([
                0x3560eec3d0531ca5,
                0xd73414ca5f7f73f6,
                0x179a6d41a7a122f6,
                0x3e95b1c35b5ff2c7,
                0x9fb4611c575a53f3,
                0x1831f118f39061fd,
            ]))],
            [Felt::new(BigInteger384([
                0xab3001466ac68070,
                0xebd5f7559f08b5f0,
                0xfe85ede14d391ac0,
                0x683f2ec7a9ed0b44,
                0x3eca747f3885d5fc,
                0x03de8ca8624e6c0f,
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
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![
                Felt::new(BigInteger384([
                    0x23a47b6b283d218c,
                    0xcc0a59365a45802a,
                    0xd2cee4f01e02cc27,
                    0x9a906932a2270b2b,
                    0x30c0953cdc966bc6,
                    0x1882d668466dabdf,
                ])),
                Felt::new(BigInteger384([
                    0xad10bfb44981e7ab,
                    0x2d4a767946cbf3be,
                    0x9a1c1481a17e5740,
                    0xe78ad746e9340a0f,
                    0x901a2cf7491dff22,
                    0x06e2370129e50196,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x924aab3df5a65298,
                    0xf4989fd072dcd4f0,
                    0x0443edf41302fe56,
                    0xb97ccb6d02107510,
                    0x7335acf7c95024d3,
                    0x07e243c83f50663f,
                ])),
                Felt::new(BigInteger384([
                    0x6f68b680af405248,
                    0xd8f7f77dddfb94ff,
                    0x3b128d01aa041a7a,
                    0x87ff1084d837019b,
                    0x6f41f47b2627d34e,
                    0x14c2858f45ba1256,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x8418600f1e154d9b,
                    0xfd9793222f27cc04,
                    0xe8d2bf5f9807863b,
                    0x7b5454c2faf75e9f,
                    0x5b528d38372c5443,
                    0x046f677d6bf19933,
                ])),
                Felt::new(BigInteger384([
                    0x4ca7f58fd38addd2,
                    0x5d129b4ae21dcb97,
                    0xf83487062d2b12ed,
                    0xc4338fd794a000a2,
                    0xa1ee4cf425fb6e15,
                    0x0b8cefb9f554f412,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x199dd46b85a43710,
                    0x0f744f9ae2de3a07,
                    0xa903eade7424bb90,
                    0x0ce1ea2524c92d5d,
                    0x58473d7a4b4be92d,
                    0x009171238ca91ad7,
                ])),
                Felt::new(BigInteger384([
                    0xf6f5b3d649e98675,
                    0x133fa03149fcd460,
                    0xd8bbe403009a5a38,
                    0xd1f05144a8f7903c,
                    0xb40958aefc6aa32f,
                    0x097f1ca79d05b43e,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x63a7241b769443aa,
                    0xd335b2705baa9dac,
                    0x4802ba8fd586bfac,
                    0x3a03b7b269104dad,
                    0x8ee7c25d63b00612,
                    0x13388f15c850bf32,
                ])),
                Felt::new(BigInteger384([
                    0xfee585c6db2db6d6,
                    0xffab2823bde0f048,
                    0x4fc6d7fc74b611eb,
                    0xc236302189783026,
                    0x9cdb5da19d8d0759,
                    0x05bb21cf7dcdc846,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xb8b9f53894adbbcf,
                    0x0351e8d30b72532a,
                    0xfaedbf451318b972,
                    0xc9363f4c53b0b275,
                    0xe13f32353c6757cc,
                    0x14c937d93d990b14,
                ])),
                Felt::new(BigInteger384([
                    0xb13e30f560b567a3,
                    0x979ed3b1c4bfd0e2,
                    0xea41fd5ca8e13cf8,
                    0xcb963916c6abdd59,
                    0x53ff7a8d2ee7d670,
                    0x15c477c32b5a415e,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x1f182aab72e4e166,
                0xf5d14db63dfa3be7,
                0x82a9a8f31750613c,
                0x59d205e5f733cf9d,
                0x34698e6670505df7,
                0x050cede801109c51,
            ]))],
            [Felt::new(BigInteger384([
                0x4f7f805a314f09c2,
                0x5ceb172fa5575e26,
                0xaf06b132231621e3,
                0xae219b1ed2049ab1,
                0x8521506bf1d902ec,
                0x0726736187681866,
            ]))],
            [Felt::new(BigInteger384([
                0x0d0163876b3d274d,
                0xe85198908cef6c35,
                0x3e07f214593b316e,
                0x072794075b066138,
                0x3e5fdaff4e4dc4ad,
                0x1791d86c412f05a6,
            ]))],
            [Felt::new(BigInteger384([
                0xed2afc8a5e9084d7,
                0x36f4bea563b43b6a,
                0x631ce3ce1c231ec7,
                0x331c00d5129ea54c,
                0xbf171f469eddc7da,
                0x1268fe912505636f,
            ]))],
            [Felt::new(BigInteger384([
                0xadfe28196faaf423,
                0xee22b05919b56e99,
                0x43afa6e72f3e927b,
                0xcd68eb574a61b767,
                0x3ce09e9b5715c4cf,
                0x03a9811184d7bc72,
            ]))],
            [Felt::new(BigInteger384([
                0xf4d9d4daa5128ed9,
                0x19b0e78875c7b76f,
                0xa19f3cb6b949072e,
                0x3dedb1e90eccffff,
                0x94571a0a4dbf7385,
                0x0ffb914955dba96b,
            ]))],
            [Felt::new(BigInteger384([
                0x6663d133957cfd14,
                0x16c16778d575055d,
                0x23a0876149ddc335,
                0xe2f23da686d565db,
                0xf1cb6ebfea2d50d1,
                0x0f1592155beb5d62,
            ]))],
            [Felt::new(BigInteger384([
                0xcc02bc9cff375aea,
                0x566f560b5f73f43d,
                0x1192b3113525f114,
                0xb7b233af2c5d490a,
                0xad51e1a27ae5bb0f,
                0x0c6e77e43860efe3,
            ]))],
            [Felt::new(BigInteger384([
                0x2c10fde92700bb8b,
                0xe719045d20010fe9,
                0x123df1d199a35ad0,
                0x3f8f49b4a0a96dc7,
                0x3d2d387ea3ac33fd,
                0x00b161e506968647,
            ]))],
            [Felt::new(BigInteger384([
                0x630aed6db42474ca,
                0x12783a3c84b83a42,
                0xf2125faef6f93301,
                0x93a569f8af5e7b35,
                0x1f21d0d326f13229,
                0x05996e2672afe637,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }
    }
}
