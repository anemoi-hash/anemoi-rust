//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, STATE_WIDTH};
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

        let mut digest_array = [state[0]; DIGEST_SIZE];
        apply_permutation(&mut state);
        digest_array[1] = state[0];

        Self::Digest::new(digest_array)
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

        let mut digest_array = [state[0]; DIGEST_SIZE];
        apply_permutation(&mut state);
        digest_array[1] = state[0];

        Self::Digest::new(digest_array)
    }

    // This will require 2 calls of the underlying Anemoi permutation.
    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        Self::hash_field(&Self::Digest::digests_to_elements(digests))
    }
}

impl Jive<Felt> for AnemoiHash {
    /// This instantiation of the Anemoi hash doesn't support the
    /// Jive compression mode due to its targeted security level,
    /// incompatible with the output length of Jive.
    ///
    /// Implementers aiming at using this instantiation in a setting
    /// where a compression mode would be required would need to
    /// implement it directly from the sponge construction provided.
    fn compress(_elems: &[Felt]) -> Vec<Felt> {
        unimplemented!()
    }

    /// This instantiation of the Anemoi hash doesn't support the
    /// Jive compression mode due to its targeted security level,
    /// incompatible with the output length of Jive.
    ///
    /// Implementers aiming at using this instantiation in a setting
    /// where a compression mode would be required would need to
    /// implement it directly from the sponge construction provided.
    fn compress_k(_elems: &[Felt], _k: usize) -> Vec<Felt> {
        unimplemented!()
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
                0x1d54be3fbfc96851,
                0x8f2d2338d617b5c4,
                0xd424e40bc541e5be,
                0xfec2340709d4b0e1,
                0x4ac7182bdaa4f2c4,
                0x00d5456a290e5502,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x16dae2b9f8530c7f,
                    0xff4cfc955edf9d4a,
                    0xa07937db85e3474b,
                    0xda279fcda6173bbc,
                    0x5b223944da7bd9db,
                    0x00bb0eef3866ba24,
                ])),
                Felt::new(BigInteger384([
                    0x7867eea189bbb79b,
                    0x1c2553c71b672d83,
                    0x849764809626cd42,
                    0xf6149f9e72929089,
                    0x644d74c6f30b9949,
                    0x012e1f73507e61dd,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xec07ad3a12cde906,
                    0xc10126d88c45ea0d,
                    0xf8416aa65e9a670e,
                    0x30599c159531a92b,
                    0xc7c2cb58169f2a12,
                    0x0157acbbc5185211,
                ])),
                Felt::new(BigInteger384([
                    0xd4759fc0b94e455a,
                    0xbf9678dc7111f0e9,
                    0xd1f773672009cff2,
                    0x86da84154380e415,
                    0x2afe23fb64bf907e,
                    0x004990029d43bcd7,
                ])),
                Felt::new(BigInteger384([
                    0x16b58d3b811c2d83,
                    0x8a07550fc3c67adb,
                    0x6437587d82fa70f3,
                    0xf500ce7e2fa5e3d7,
                    0xa04000329fc8274c,
                    0x01372c2259ce104a,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xc2429f628e24157f,
                    0x23e8c952d2e7d992,
                    0x4f0fc81d6a78925f,
                    0xbc35e75fc4ae530f,
                    0xed662185c6a6c9ae,
                    0x00eb964863418ec8,
                ])),
                Felt::new(BigInteger384([
                    0x013602c6214af987,
                    0x46050651765ee119,
                    0xb37b499fc9eec83f,
                    0xcd36513e74cecd61,
                    0x284b61bc7b009c2a,
                    0x014b5cef423d52c9,
                ])),
                Felt::new(BigInteger384([
                    0xb88f281b8794e5d5,
                    0x0c0d3a63bf2f5dbf,
                    0x89213a5fbda0587d,
                    0x5244ab60c8cd744b,
                    0xed72f5e42588e2ee,
                    0x017beeac11d20989,
                ])),
                Felt::new(BigInteger384([
                    0xfca6723feec867a3,
                    0x0b55bc8b5f2a1b6b,
                    0xec669d23ed18c318,
                    0x1cee606a862c8b1d,
                    0x20b789aa296aecb0,
                    0x008650cc3c196204,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x5045ebba10965af1,
                    0x63982a8fad045790,
                    0xafb33455b2a99f31,
                    0xf43f7c7b07f1af0d,
                    0x9e88f22d4dbb18d7,
                    0x0144612dc83c1542,
                ])),
                Felt::new(BigInteger384([
                    0x9b193a69372b3067,
                    0xb83bf79284ef75e9,
                    0x6d5b9b33e71ec540,
                    0x4787a7f9256a544e,
                    0xf0adcb8914ff6332,
                    0x00bca71cdf570d8b,
                ])),
                Felt::new(BigInteger384([
                    0xd5c6764f2d7d817c,
                    0x65dd81a588e13abd,
                    0xacd12ac57666d669,
                    0xad7e34cbe851c317,
                    0xa295a72da295a921,
                    0x000d0d4a85146b45,
                ])),
                Felt::new(BigInteger384([
                    0x0fd1e91b594d4a56,
                    0x1b7b321e94aa2680,
                    0x0e2ca5d957c62eb7,
                    0xf9eaa52b9b3c1f5f,
                    0x80290cd5af67076a,
                    0x010ed85b99302bb9,
                ])),
                Felt::new(BigInteger384([
                    0x0812d119a659fc6c,
                    0xaa2a1463d519aa38,
                    0x9f5253ab1c9584be,
                    0xff7bed95f48c21d4,
                    0x5eece7acc6e0dc24,
                    0x0112046fa6ef3649,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xd55188ec4139c570,
                    0xbf8fce8e4a120353,
                    0x89104164a5f9884d,
                    0x3ebd58a9b45e45fc,
                    0x2c52dba6c176c34b,
                    0x0021bfa35beda0bf,
                ])),
                Felt::new(BigInteger384([
                    0xdeb58f43c983c733,
                    0x8929cfca7a5727ff,
                    0xdfe743c7053face0,
                    0xfedc278dafc844ca,
                    0x8a598e3635dcda34,
                    0x016f705b0b2bd4d3,
                ])),
                Felt::new(BigInteger384([
                    0x308be39c3768f5bf,
                    0x10f9a0cd48f9b82f,
                    0x69ff817bac20e54d,
                    0x8c9685624a5f437a,
                    0xcdab00c3909e2ecb,
                    0x00ef3b982202dfaa,
                ])),
                Felt::new(BigInteger384([
                    0x3be663334aed091f,
                    0xf607463a852f750c,
                    0x5ffb27f6be024c60,
                    0x3fb0c0732fbed6d3,
                    0x17e4526c75b4f05b,
                    0x0075b7ab9098fba9,
                ])),
                Felt::new(BigInteger384([
                    0xe5bb82e896ee3f09,
                    0x42d7d4178b504f83,
                    0x746605c99f068a10,
                    0x104ff1901cfa1b40,
                    0x87ab51a1ea565a96,
                    0x010e962e7f652e64,
                ])),
                Felt::new(BigInteger384([
                    0xe4592d8fc326c815,
                    0xe671294dbbaf7f1d,
                    0x3cc249fbeb4ea506,
                    0x3224080fb6e741dd,
                    0x2175117c787ad8e5,
                    0x0138ed00db8c2de4,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0xa340087482dc6b03,
                    0x92488eb3194c68e7,
                    0x544d86105472090f,
                    0xb157de9d03b0faa1,
                    0xde27aa7ca682eb2b,
                    0x00052611bd3bfe55,
                ])),
                Felt::new(BigInteger384([
                    0x29ef5c51daf46e8f,
                    0xf683ea11ad90e1ec,
                    0x800a0500553d3466,
                    0x8c83eb20702e6b60,
                    0x4546894a76b6f995,
                    0x016f75d01eed77ba,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7a7b7167c3b7c99e,
                    0x07066ce7cd9ffe4f,
                    0xcdda324ae65022ab,
                    0x295b26922d23543f,
                    0xd9f4697abb3f59d1,
                    0x00ed4361998c84d8,
                ])),
                Felt::new(BigInteger384([
                    0x43e253eecac1105f,
                    0x261fa36832e4b38d,
                    0xa726688b5f3bb59a,
                    0x17c41eed740485c5,
                    0x9b02fb249a4dd229,
                    0x0101c9c847b40b0d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0d42c09f76268fdc,
                    0x4ef3e3d853052a39,
                    0x13562b85f635bcbd,
                    0xf3d24f66b20f6317,
                    0x54668fdae796dbd2,
                    0x007781003f93ff88,
                ])),
                Felt::new(BigInteger384([
                    0x531daf20b2157e4e,
                    0xd77dd929013f224c,
                    0xbe35f9bd325e1e40,
                    0x6dd34a9f6b9260bd,
                    0x59069839784650bd,
                    0x00e4c2510747bc41,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x833f76c2667c9930,
                    0xa14a3ed15183eeba,
                    0x63a97e8f77ed3eff,
                    0xb72bded87db55ee8,
                    0xb7834a8d87f09d39,
                    0x005a364388e48440,
                ])),
                Felt::new(BigInteger384([
                    0xae14ccec8ff1e8dd,
                    0x523c1e6f3fe67de3,
                    0x4f503f2e7981887e,
                    0x1a045aed1d9f0001,
                    0xa109197519fc9a53,
                    0x00236edbf36d3978,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x15c49ebba2373152,
                    0x26167682bd2c737c,
                    0x546b8e28b653f303,
                    0x2574cae2942f3322,
                    0xe5a588b65e586091,
                    0x0148db5cb97eff60,
                ])),
                Felt::new(BigInteger384([
                    0xebc10c8009b7584e,
                    0x5cf9885fa87df880,
                    0xbd8ff90bcc57161f,
                    0x05ca21c23ea9892f,
                    0x9f07a6e628a0d41a,
                    0x00f338e60a221b12,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x393010d28c6dd843,
                    0xc1793af360faabe7,
                    0xc681dd747b8d0423,
                    0x84da9120624b8fbf,
                    0x9c42ad9f279305ee,
                    0x018bcbfe9ae18dd8,
                ])),
                Felt::new(BigInteger384([
                    0xe864c19ec37f2476,
                    0xd7588a44bc96545b,
                    0x2c3a2962901cf5ec,
                    0x8f53c92132ec0392,
                    0xf6eb64611aec9475,
                    0x01258834fa8661dd,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xee2a96bce8d9875e,
                    0xfb09ad97fe5bb198,
                    0xa5b4f762b6703950,
                    0x6e823c0a3ba687f4,
                    0xf4e83d4aca3d6e1d,
                    0x005079a0742c7bd2,
                ])),
                Felt::new(BigInteger384([
                    0x8fd42a1b7267f2f3,
                    0x936a9700e22c05e7,
                    0xa04396eb459da69d,
                    0x96342edbc474fb2e,
                    0xc7a91a07f4d9b076,
                    0x006adc461efb297b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb5d279a143716f67,
                    0x422387b381a7b855,
                    0x10f357f2b6f12a11,
                    0x0efb20ccb7dfaf05,
                    0x525cf7231dbbc244,
                    0x00aa73cbeaa28978,
                ])),
                Felt::new(BigInteger384([
                    0x485269f38f1b7519,
                    0x696e04c70c10bb00,
                    0x32a020cd493fa377,
                    0x4253f59a6c038b3f,
                    0x29e3e7f11adf2a7f,
                    0x00fa4dc5cbe6057d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x895dfb4a785143d7,
                    0x5a5abff85db6171c,
                    0xf00ae1e30c3c8b30,
                    0xcb5000ba6531fd67,
                    0xda0270341f4078a2,
                    0x007046a4e1174eff,
                ])),
                Felt::new(BigInteger384([
                    0xb5317432125bb39d,
                    0x8aa09508bd44b071,
                    0xff253bd9e3d6bfa6,
                    0x15568e57f24e9140,
                    0x6743cb57a2b7e930,
                    0x0058041b9156cdf1,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x53a6eaf43ed7ddd6,
                    0x3f2d9d3e54024bde,
                    0x7ed091e5adc1693a,
                    0x6be2d791de2c0e51,
                    0xb5456a668608f8de,
                    0x00fbcc07a1f1a418,
                ])),
                Felt::new(BigInteger384([
                    0x15750a24f07ebcb7,
                    0x2ea0fdc08af83b83,
                    0xb4580224b1938b05,
                    0x22d645f97d2fee2a,
                    0x13999fb70b76d602,
                    0x0113e124d30a9fba,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }
}
