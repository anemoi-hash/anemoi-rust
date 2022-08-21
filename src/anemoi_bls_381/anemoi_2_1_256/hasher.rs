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

        AnemoiDigest::new(digest_array)
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

        AnemoiDigest::new(digest_array)
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
                0xf6e85e4a6c05052f,
                0xa3e9f05a7b2abd71,
                0x38912189bf0ef6f4,
                0x3320b347d36f0849,
                0xc3ef58cccda591c2,
                0x182b931659d7a1f8,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0xa99e97dc0d2d82b9,
                    0xf3ca131ca69521f5,
                    0x18d5e25bec3262a0,
                    0x8dbdc2392ee195eb,
                    0xcb519d167c9ba21e,
                    0x18144ad47255be49,
                ])),
                Felt::new(BigInteger384([
                    0xbfbed83d879987de,
                    0xbd65930b6612747d,
                    0x37d8d5b1986b399a,
                    0xdb579c1f25672c2a,
                    0xe683313d6fdc853f,
                    0x060592151fa1346e,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xae0eb748b638ddf8,
                    0x9a4785c6348aea89,
                    0x8c2736c7999353d7,
                    0x1fe66d07436e3bba,
                    0x5e65608817a27a6b,
                    0x1759e8c4eb954962,
                ])),
                Felt::new(BigInteger384([
                    0x3100be22d5a2c555,
                    0xe8ecc1e54712a66d,
                    0x96a9603420af84c0,
                    0xf7eb6a72a74032cc,
                    0x3136433ca3ba3ab1,
                    0x0623a5ec0c5ec0e9,
                ])),
                Felt::new(BigInteger384([
                    0xb738e463faf11f74,
                    0x6da1484c443aeced,
                    0x8bdc817e57388093,
                    0x0f24d848574a12bd,
                    0x489953c453a60f18,
                    0x0ad08cbbb613cf68,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x3af8e9fb6aa70814,
                    0xf362c300f23ba5a2,
                    0x18b24b555d7c7c71,
                    0x70483bd7a1dbbe30,
                    0x650b16a2641a3484,
                    0x14949ce6a9551158,
                ])),
                Felt::new(BigInteger384([
                    0xb03f8d13ca07f067,
                    0x56e5642b9896e962,
                    0x51a9c7768556d140,
                    0x8a40ca4086660ce0,
                    0x84294108f18c245c,
                    0x17bf3949e9bd5ba3,
                ])),
                Felt::new(BigInteger384([
                    0x59c8d6f59559efe2,
                    0x0d1b37a2c4db216b,
                    0xffba09f9dcc0cceb,
                    0x193120db55acdaa1,
                    0x9b5bfa4285a06381,
                    0x0474897862fbfd00,
                ])),
                Felt::new(BigInteger384([
                    0xdf3c5b9111f5191e,
                    0xa109cf7895b2177f,
                    0x0b1911b13ed5f1bd,
                    0x4a719a8b67a83f98,
                    0xf0ff864958380a0e,
                    0x07ab8b4099f675bc,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x9208c5a17ba0e934,
                    0x6ff9970b6c3ec446,
                    0xdd4f83aff1ceb3b0,
                    0x1393b5312e74da5e,
                    0xd073b96da93dcc54,
                    0x07cba019ab1b1a31,
                ])),
                Felt::new(BigInteger384([
                    0xa10a34c25192616e,
                    0x9150043a3853aac7,
                    0x54a5c3b0c001c873,
                    0xafcef11627152181,
                    0x9ee4b22e7b3ac716,
                    0x10b9b87bf6509fc0,
                ])),
                Felt::new(BigInteger384([
                    0x5e4870b39b77fe40,
                    0x9bd41e8fa40b8af2,
                    0xd97dc8723fd12c00,
                    0x4e70e0b61e45b329,
                    0x8f2792a6b7ad127d,
                    0x16e0bb4bed20a6f2,
                ])),
                Felt::new(BigInteger384([
                    0x6ef72781fac09ef9,
                    0x8c37cbdadf0709bd,
                    0x9b888278265015e6,
                    0x596aa6ac801a1427,
                    0x5b103caee68bf25d,
                    0x0990180fd9a5e9ee,
                ])),
                Felt::new(BigInteger384([
                    0x9f3e1d5b3f07e60e,
                    0x357e778ea3b0cb67,
                    0x60491e0bbc0028c2,
                    0x90668fb4034ada14,
                    0xf120914b902d427b,
                    0x0bf004846a8af2d6,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x5932ab32a876750e,
                    0xb85f4c6e7ed8ad42,
                    0x4a4f44bb3ed472a2,
                    0x1fb0ca6ae9fa95b0,
                    0xe06d73fa0efb1f79,
                    0x0dc617d7a76c2035,
                ])),
                Felt::new(BigInteger384([
                    0x57b5e22c17d4c0df,
                    0xdcf811e9d57d5451,
                    0xb789809199a07ec2,
                    0x142ff1765d6ca73f,
                    0xe5502b86421cad14,
                    0x093ee298874c2537,
                ])),
                Felt::new(BigInteger384([
                    0x67792f8960fe81d2,
                    0x102f24351499dad4,
                    0xce410fe5f39aad58,
                    0x1b6fd55ad66194f6,
                    0x94d4b331a392816d,
                    0x0f1a98230ee6c05a,
                ])),
                Felt::new(BigInteger384([
                    0xd5b8f9701dbee537,
                    0xced2a56810cc1e83,
                    0xa52a111342c08e55,
                    0x9be0e467a459d697,
                    0x36939f044de13370,
                    0x141a6ddf0f5958d4,
                ])),
                Felt::new(BigInteger384([
                    0xdfe5a527d3130952,
                    0x844fb59e2af81080,
                    0x6051e7730231f4bc,
                    0x65dc39d79a100ebb,
                    0x52a11d5afefd33a5,
                    0x1653c12ed2c6528b,
                ])),
                Felt::new(BigInteger384([
                    0xc7c11a4509c96f21,
                    0x16ba4d24c9fbd5af,
                    0xe0c1df20cfe98b3e,
                    0xd75eb4be1bc4d1b4,
                    0xb2f12fc174f8a9d7,
                    0x08bcbeb1d3e167a8,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0xc1187a38435d94c7,
                    0xebadd805789c88c0,
                    0xee89ae5b6e220d7c,
                    0x0c69af0db52ccab9,
                    0x9bf758315a72f075,
                    0x10d3e0d43ce31bdc,
                ])),
                Felt::new(BigInteger384([
                    0x71fa33893fdaf9b9,
                    0xaeb67dc486b85fa3,
                    0xcae8807a5449a7f9,
                    0xd97edcbffdb8bd4e,
                    0x9b6fac2327a74279,
                    0x17c82aa384d85a07,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe9a62e60d2d0860d,
                    0x80fe938170aaf374,
                    0xffd8e17a010f507d,
                    0xf28b2e42b704e102,
                    0xd1811b7c2a9ecac4,
                    0x174e619d7292ef90,
                ])),
                Felt::new(BigInteger384([
                    0x2b8a86306354ff9f,
                    0xe6147e2195c81987,
                    0x5d69b5c4c19fd94f,
                    0x19f46ca31adbf58e,
                    0x52a0ced4595dd1a8,
                    0x009c374102e81e87,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7471adb474c82d53,
                    0x59812f11c62fcbac,
                    0x74dfbd62d5958d7a,
                    0xe0baaa84e4746637,
                    0xfedd79583523c091,
                    0x118cda4010f70228,
                ])),
                Felt::new(BigInteger384([
                    0x6600f3f4ac4b8282,
                    0xd80d4b039e7f1fb2,
                    0xc56efb11a23231ec,
                    0xf5cf5455d7841a76,
                    0xbb458413ba63dfc3,
                    0x00f464b73da263c4,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xba78943231698488,
                    0x91313e8f32c04d79,
                    0x337bcfc8254015f1,
                    0x7267f62edb0fd7bf,
                    0x29af08a7de1111c6,
                    0x0491c900a29859a0,
                ])),
                Felt::new(BigInteger384([
                    0xbc9101d53e58e083,
                    0xb2561d48e7fd183e,
                    0x530eb93082240856,
                    0x1df0188e092c45a3,
                    0x5807cbbfbda6c35c,
                    0x0c53d05014abec6a,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb09e8c60ed0bdd57,
                    0xec0e1813fcdc4eda,
                    0x9797fef75c102386,
                    0x51cb627f993b7df4,
                    0x2b485c24385aeb4b,
                    0x0b52ddcdd629f438,
                ])),
                Felt::new(BigInteger384([
                    0x02bc6dc11135bb18,
                    0x6830385c73323ffe,
                    0xd86693f52b9295ba,
                    0xcda259cbc502bb97,
                    0xa77bd720d40bc47f,
                    0x1202c0e346ff6485,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9091e5e694220d47,
                    0x637b5b63bb2bb87b,
                    0xd45e0fd2680ca6a1,
                    0x91941a0745b6b6ed,
                    0x6105710a60c81367,
                    0x05e34f25ca8c55f5,
                ])),
                Felt::new(BigInteger384([
                    0xdd168bcc4a041311,
                    0x937e694636d9aa83,
                    0x53d5a0c633a51e02,
                    0xcbe858646cc9c09f,
                    0x887dc461fcf139ab,
                    0x013fe43f084cf2c2,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xf104cda323d37b29,
                    0x7930f0645633e269,
                    0xf3da8e7f2ea096dd,
                    0xa360f921e4d3f1f6,
                    0x40167b56b1bc882c,
                    0x152567f3011e27cf,
                ])),
                Felt::new(BigInteger384([
                    0x9ee68d1b3f86e0e5,
                    0x7025fa3dfdc6dd1c,
                    0x73158e9c0b8e57a9,
                    0x6515fa64a5f84b97,
                    0x6991a155317b27e9,
                    0x1026d674d479e372,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6c56fff22173b477,
                    0x5f17bbdc1aa33c74,
                    0xd7494b0e1edbecde,
                    0x5e45687a82b4c168,
                    0xa251290e9d5e103a,
                    0x026b36926b4541d7,
                ])),
                Felt::new(BigInteger384([
                    0x8f9689ce629b1c5b,
                    0xaa92e741140b445e,
                    0x3ee22e25149df65e,
                    0xde1185b48284bec9,
                    0x40ba0659319f17ac,
                    0x109e76668ac6876b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6f84bc881953b2d1,
                    0x3a62baf479210e3a,
                    0x2a34bd189415f028,
                    0x08e607ed757a283e,
                    0xe1bcbb1f624b22a6,
                    0x100f4364661a16a9,
                ])),
                Felt::new(BigInteger384([
                    0x1158d3a5487f0390,
                    0x1fc42fc3480d0520,
                    0xcc5a408d3c5fba87,
                    0xd74c0ce94d02d92b,
                    0x50e026b17796167a,
                    0x116ca7f54e411c83,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb0c717789981b4b0,
                    0x53202bb3b8bb2c92,
                    0xe369fbc6dec58eab,
                    0x0b4f0c210aac72bb,
                    0x3f75fe3154b29225,
                    0x14c1a7d259506fa9,
                ])),
                Felt::new(BigInteger384([
                    0xbca6de34feea40be,
                    0x4461642d293523c7,
                    0x0e295282302a6718,
                    0x4ec2ac5d55a59d36,
                    0x9add01d38c2ccbc0,
                    0x12088163ec128cdc,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }
}
