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
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 32];
        for chunk in bytes.chunks(31) {
            if num_hashed + i < num_elements - 1 {
                buf[..31].copy_from_slice(chunk);
            } else {
                // The last chunk may be smaller than the others, which requires a special handling.
                // In this case, we also append a byte set to 1 to the end of the string, padding the
                // sequence in a way that adding additional trailing zeros will yield a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 32];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 31 {
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
    use super::super::BigInteger256;
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
            vec![Felt::new(BigInteger256([
                0xfad610bf5bc9c9d2,
                0xb25fb3b2d1ec0d44,
                0x6671726dd7f27624,
                0x0f064fb6d78a906b,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x5407df653da2cb46,
                    0x1ea52196055dc82c,
                    0x4f487a0f1de38147,
                    0x1eab8516945b0dfe,
                ])),
                Felt::new(BigInteger256([
                    0x8f4f4b238ca76b01,
                    0xdd8833ff53fb3579,
                    0x9da47805873557f6,
                    0x02aee8a4279aad6d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6782bc0f36871667,
                    0xa22896f557f1eb6a,
                    0xd0d0b9b52898b57a,
                    0x1e86b8481a416c78,
                ])),
                Felt::new(BigInteger256([
                    0xb1037f0196ae6145,
                    0xdf4a8cb6d2edeec8,
                    0xbe8fd1c5703ecd24,
                    0x1c235873f7b368eb,
                ])),
                Felt::new(BigInteger256([
                    0x62919e8ac6cc604b,
                    0x662526d6e2ca996c,
                    0x0a9ce2c16d303039,
                    0x2b7589fbf9c397ad,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x82284308d5c09ebc,
                    0xd61cbe821015a6d7,
                    0xde60590a48ee9df9,
                    0x14197bb4635633f0,
                ])),
                Felt::new(BigInteger256([
                    0xe79fceff88fb3c69,
                    0xec29d40e3cc5949b,
                    0x9405b7fe7e23867c,
                    0x1a57f36429b98acf,
                ])),
                Felt::new(BigInteger256([
                    0xadbc234243fe332f,
                    0x75e11379724726d0,
                    0x202a3ba8925b962e,
                    0x0e075eff4cca8dda,
                ])),
                Felt::new(BigInteger256([
                    0x24c139cd7074be9f,
                    0x5a2fdae39256368b,
                    0x0beadee32360f6fb,
                    0x287346cc8252bd05,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3fd47ffeae64b8c7,
                    0xfc4b9c5d0c508908,
                    0xb922222524812408,
                    0x0546d5337b2f4c0f,
                ])),
                Felt::new(BigInteger256([
                    0x8fbbea7ac293fb27,
                    0x0afdf7982b6c7bbf,
                    0xc8b3c0c78decf426,
                    0x0afe7edeee80eaf5,
                ])),
                Felt::new(BigInteger256([
                    0xe07c25b662ce9f64,
                    0xcbeb955c92fd2430,
                    0x7c8e9408d26f3e14,
                    0x2ede62bd9a0e771a,
                ])),
                Felt::new(BigInteger256([
                    0x2510dcc3d32e9e32,
                    0x48d8f43b8132db64,
                    0x329bfd7c9f558cd9,
                    0x063a284b3e40426c,
                ])),
                Felt::new(BigInteger256([
                    0x6ca6ad45f508def1,
                    0x658e9716a51b54ea,
                    0x0c75044162c13b49,
                    0x1bf695769927e599,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa5c9ff35aee38166,
                    0xf45ee3289a07c45f,
                    0x1c264eeebc52a977,
                    0x238417e59cca8c9f,
                ])),
                Felt::new(BigInteger256([
                    0x7566293c0772474c,
                    0x465492a740a75819,
                    0xa8383de205ebeedc,
                    0x2daf91a87368eb78,
                ])),
                Felt::new(BigInteger256([
                    0xe8bf42d2d0330d15,
                    0x850553744ac4a622,
                    0x401e882f5d29d61e,
                    0x24426e8b7624c923,
                ])),
                Felt::new(BigInteger256([
                    0x28abf640b575705a,
                    0x2dce07b12aa9f74c,
                    0xc0b7bbaddbe7b0c6,
                    0x0b14a6a6bb1cb259,
                ])),
                Felt::new(BigInteger256([
                    0x4984ecf951cb0571,
                    0x37d938448392bea4,
                    0x8424b26766cf3bbe,
                    0x0bb29590901417b8,
                ])),
                Felt::new(BigInteger256([
                    0xd730a03a5b23aaae,
                    0x389a5a7a58a974e1,
                    0xb87f67ee82931b4f,
                    0x20aae5215a567ddd,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x12073697ba1b54cd,
                0xa748eb3a0ec8bab5,
                0xad3c6771ab998832,
                0x2b403cde8964d7d2,
            ]))],
            [Felt::new(BigInteger256([
                0x38d5521d135a6e5d,
                0x75e9eda46411d63f,
                0x35a4d8a31f0e5d6f,
                0x12018c84afd0c1ea,
            ]))],
            [Felt::new(BigInteger256([
                0xa7a72ebc0c4e555d,
                0x3dfa38e339c67bb3,
                0x4dd4c5bbace6b292,
                0x02c9c235eb90da1e,
            ]))],
            [Felt::new(BigInteger256([
                0xf883bf5c8212310a,
                0xec0c11ffe27f91bf,
                0x2f6dfeb6c4740c4c,
                0x1a26ec99a836580a,
            ]))],
            [Felt::new(BigInteger256([
                0xa83ad91a0958179e,
                0x526cccd3f98081fc,
                0x72660ed217baff8d,
                0x12ce0d9ca0a4000d,
            ]))],
            [Felt::new(BigInteger256([
                0x3a22d4928556734e,
                0x12b87966c573babf,
                0xa6990787b0a6e07d,
                0x2e541d0db72ec11f,
            ]))],
            [Felt::new(BigInteger256([
                0x513b78fe4118c4ff,
                0x665cc53d1b36fbd1,
                0xd01647a0bb530fa5,
                0x278cc83fa0bd738a,
            ]))],
            [Felt::new(BigInteger256([
                0xd47afa219c8824a4,
                0x970cc8e7c1eac5a7,
                0xbb52337d1f7455fc,
                0x14c4eceec2605341,
            ]))],
            [Felt::new(BigInteger256([
                0x283ba8f64c1b4b58,
                0x714fc40e3a257426,
                0x7acdc2be061c2335,
                0x11b13c75b8123551,
            ]))],
            [Felt::new(BigInteger256([
                0x19f1be10e72dd119,
                0x98438ef3f7952747,
                0xa99a05958d15a527,
                0x18477be673a96e74,
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
                Felt::new(BigInteger256([
                    0x307c5c45892c4b27,
                    0x7af2075438ae58be,
                    0xc25723271ab1f09a,
                    0x0995343aa4adaab8,
                ])),
                Felt::new(BigInteger256([
                    0xcd2c4bd32b27117c,
                    0xb20a186f1ebb17a2,
                    0x9724ec00c1e03328,
                    0x101b4fd7b002945f,
                ])),
                Felt::new(BigInteger256([
                    0x8dad4195476ad2b5,
                    0xbcf02d67781e1320,
                    0x1620716658c49f0c,
                    0x14ec2a4b876ded7f,
                ])),
                Felt::new(BigInteger256([
                    0x89e1e0ea7d3ceef9,
                    0x32d6e92f56eb7833,
                    0x022cb715661ed417,
                    0x0ccda2830de1f3d9,
                ])),
                Felt::new(BigInteger256([
                    0x93d81881052c3beb,
                    0xe250c9dd0318bdb0,
                    0x6c0702b45545035b,
                    0x12e7d54687387a20,
                ])),
                Felt::new(BigInteger256([
                    0xafe6fd9b1a22cc6c,
                    0x1a2d845e551d198a,
                    0x9fe54432abe6894a,
                    0x103848fa4fda6400,
                ])),
                Felt::new(BigInteger256([
                    0xd4b5ebbfa104abd6,
                    0x2c550e52177ac935,
                    0xaabd3a29d4810888,
                    0x3030e97c80bf8e64,
                ])),
                Felt::new(BigInteger256([
                    0x2962bf426f52f3e1,
                    0x51662b04ee48d470,
                    0x8e48cd4e72038004,
                    0x14ff876bbfd48de5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2c169a67afdf71b6,
                    0xbe709f380aa51485,
                    0x0876c58f8462ea17,
                    0x000f2f59c1f48b59,
                ])),
                Felt::new(BigInteger256([
                    0xb6e7eddb43ee5f98,
                    0xa4bdeb04351462b4,
                    0xa43a10089df3bd18,
                    0x1a89d131059725ed,
                ])),
                Felt::new(BigInteger256([
                    0xeda789ba29335de9,
                    0x5cd530f31b74f907,
                    0xa50786b6b8c608e9,
                    0x22c6eff9e9ce6827,
                ])),
                Felt::new(BigInteger256([
                    0xe4cfcbbcfbb5412b,
                    0x5c4b361ce55492bf,
                    0x3866943b9d00fa25,
                    0x00063401f3b428c0,
                ])),
                Felt::new(BigInteger256([
                    0xbc51e69535ac4fa8,
                    0x6733ccbef36f0aeb,
                    0x1f4c6224abe890ce,
                    0x17614bd62796401a,
                ])),
                Felt::new(BigInteger256([
                    0xd1bd698b7995bc8b,
                    0xc172096e9e0dd212,
                    0x62bf95505b008c9d,
                    0x0f390f83659f34e2,
                ])),
                Felt::new(BigInteger256([
                    0x045d6410cb09ac59,
                    0xdf8df7696f12e908,
                    0x99bcc392dc1a166a,
                    0x269662ab5722aada,
                ])),
                Felt::new(BigInteger256([
                    0xff569930b8cbed0b,
                    0x85bfdbddf67c8448,
                    0xe78bdaa89c148353,
                    0x0f2cd8334688afcb,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0c451a78db214315,
                    0x28d4531676689dfe,
                    0xa84e45f72823857e,
                    0x1ea6b6f4063e992c,
                ])),
                Felt::new(BigInteger256([
                    0x87552ee5507e7f31,
                    0x4d4c1cd308262bb1,
                    0x8c6659fa79e02ce7,
                    0x1b1c5510a7f6d85a,
                ])),
                Felt::new(BigInteger256([
                    0xf5498a9a7155cb43,
                    0x582dd1a3932939ba,
                    0xe6035d7a85933ec3,
                    0x21cbbe1b01bd291a,
                ])),
                Felt::new(BigInteger256([
                    0x13b9401da7889ee0,
                    0xd0ef14aee278cb0b,
                    0x2f41e18f3c1abcb7,
                    0x0ddf12cb985a0a88,
                ])),
                Felt::new(BigInteger256([
                    0xbf19011064386133,
                    0x546cabda3ae488dc,
                    0xbef53ff5a0a1d036,
                    0x28d30152ce1a4107,
                ])),
                Felt::new(BigInteger256([
                    0xeca7aacc226cbce1,
                    0xe64eb8f3bf4d7c91,
                    0x3164504c775429c0,
                    0x24c2979f364ba00b,
                ])),
                Felt::new(BigInteger256([
                    0x81ad1bb4635d0aea,
                    0xe36ea42efa823be9,
                    0x13fc5612260c7c91,
                    0x08a2c1d67036c3ff,
                ])),
                Felt::new(BigInteger256([
                    0x78bb209bb31d7769,
                    0x25a3bf720021a54c,
                    0x4520edc85d655f3d,
                    0x0e216aad4994ade6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4bc21ab10184d792,
                    0x3958e25888756e24,
                    0xc1e14596437c2d06,
                    0x23e1eee89ec6c9b1,
                ])),
                Felt::new(BigInteger256([
                    0x41d1be94304ee5ab,
                    0xb67b4e07653df835,
                    0x7c0010c881335d16,
                    0x2cd9b59e8752456c,
                ])),
                Felt::new(BigInteger256([
                    0x4da072bdbbafab18,
                    0x669b013daf79df92,
                    0x9a715dbc278e8938,
                    0x25330981b2423db0,
                ])),
                Felt::new(BigInteger256([
                    0x074c703748046daf,
                    0xac36ee0039040fe7,
                    0x2e0e4cfe3ff38c4f,
                    0x238d2a73cc5268e9,
                ])),
                Felt::new(BigInteger256([
                    0x0734805fc0357980,
                    0x7c5390c3bb6c4710,
                    0x6570e00c8105317f,
                    0x20db4fad5c96f102,
                ])),
                Felt::new(BigInteger256([
                    0xad03056e07f4ba76,
                    0xe8a88afb6fbac131,
                    0x4467aeeb31c63bcb,
                    0x11906a1dc5d9b0a6,
                ])),
                Felt::new(BigInteger256([
                    0x2cf83052b8e1d169,
                    0x2a7a4a507fdcd79b,
                    0x23ae087a38451769,
                    0x24618e72df14eecc,
                ])),
                Felt::new(BigInteger256([
                    0x8f4ff54b94045152,
                    0xb5d30640b01ddbcc,
                    0x82ddde2fd7a2b89e,
                    0x23bd689259476351,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe6e0a84b7cb023b3,
                    0xe4f750903f9e1028,
                    0x4d60478d0dccdc42,
                    0x1d521b78d931f02a,
                ])),
                Felt::new(BigInteger256([
                    0x35f195095ffdf651,
                    0x9377450efe234d02,
                    0xc1634c6ab2dd8b0b,
                    0x05cd05c84366f289,
                ])),
                Felt::new(BigInteger256([
                    0x3b52854629894de3,
                    0x34ad2680660b2eac,
                    0x7635cf2bd4cda0e4,
                    0x21a557398f41d136,
                ])),
                Felt::new(BigInteger256([
                    0xcbebdb040b22f3c2,
                    0x66b253f4f3d4454d,
                    0x879d5dc82b3d6175,
                    0x04222fec7f30171e,
                ])),
                Felt::new(BigInteger256([
                    0xdc3587387aeec2ba,
                    0x7ff03710f1991620,
                    0x19a84bfce1bee661,
                    0x08ffa644ea8c7998,
                ])),
                Felt::new(BigInteger256([
                    0x1338df65e8db7139,
                    0x6081de27a47ed3cb,
                    0x12ef41d3a645e47f,
                    0x1ad6d88b5c4c0451,
                ])),
                Felt::new(BigInteger256([
                    0x5b78641a899c2ab2,
                    0x03bd6640ccbb9e98,
                    0xabe19054147962e8,
                    0x081a64297bb80fff,
                ])),
                Felt::new(BigInteger256([
                    0x0d7c8cd8df14d72f,
                    0xad008b230a679468,
                    0x920818e1042a08c9,
                    0x28601265c8cf73f0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x597fe9f76662a4db,
                    0x9c41d2fbe5722986,
                    0x40a649011f0daea7,
                    0x20926c329e3588fb,
                ])),
                Felt::new(BigInteger256([
                    0x44ca7af8ca168632,
                    0xa40801c0aa094fa4,
                    0xf321214c2f1fa68c,
                    0x05235cdaff3a03db,
                ])),
                Felt::new(BigInteger256([
                    0xf309f10f328b22da,
                    0x516e454636899614,
                    0x6e94725fc3e7c765,
                    0x05953964b98633a1,
                ])),
                Felt::new(BigInteger256([
                    0x808777c8a69a082b,
                    0x4a8da851ef748e05,
                    0xbf77725f737daae4,
                    0x0a76a6118436c36a,
                ])),
                Felt::new(BigInteger256([
                    0x694ffd13fbbffed3,
                    0x8350ef94984a0ae1,
                    0xd34996f4bd61d966,
                    0x0c9a05db427f44aa,
                ])),
                Felt::new(BigInteger256([
                    0x9845f1a8edc501d3,
                    0x074038de0c5c69ca,
                    0x0c2a40611ca2aadf,
                    0x0512a269040b5190,
                ])),
                Felt::new(BigInteger256([
                    0x039e063941d3b50a,
                    0x1d6c2dd2d037799f,
                    0x2e9e29494adcc8be,
                    0x1ea78e47c37a2456,
                ])),
                Felt::new(BigInteger256([
                    0x0fb4c4af7e420d4f,
                    0x23714084065faddf,
                    0x39c0dbbb9c2ec6a7,
                    0x0cfb4f9bdf0cea58,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x2ab0ce0aaccc792e,
                    0x9b65d9cfb2318e78,
                    0xd7e7cab1e70e8f1a,
                    0x06913a912b0bc3e1,
                ])),
                Felt::new(BigInteger256([
                    0x51f9bfcfc47fceb8,
                    0xfec509f0def6b6c6,
                    0x8b3c92d89e5e9c87,
                    0x13333f94c3a14017,
                ])),
                Felt::new(BigInteger256([
                    0xbf7cad1480191249,
                    0x2dfb1a22424103a9,
                    0x32496eea0c4e87fe,
                    0x27a150be1c6ce3d1,
                ])),
                Felt::new(BigInteger256([
                    0x470eca2f74d274a4,
                    0xbfec503926cd1b1e,
                    0x4b222f9d6497228f,
                    0x045ca095bf5e8496,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf78ae8d316eee239,
                    0x02c651e866f2acd7,
                    0x2da38e8b174bc86a,
                    0x210804ac275992b2,
                ])),
                Felt::new(BigInteger256([
                    0x641b51a4d81ac25d,
                    0xacea517251934845,
                    0xe53d25194ebb65d0,
                    0x1db4b57b8190f74b,
                ])),
                Felt::new(BigInteger256([
                    0x8764d6ebc4ade4b3,
                    0x0a0fade4b7dbad7d,
                    0xfbe9f295ac52fd0f,
                    0x064daff605c6008e,
                ])),
                Felt::new(BigInteger256([
                    0xcc4f0ccc53c820c1,
                    0x80bd89ce8d548cca,
                    0xf5b0fc91d35ddccb,
                    0x21106d9d2a2b9504,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd6a4d13a968a614c,
                    0xb173eb7ceea4eb6d,
                    0x65a40ed442f43c4c,
                    0x0c677c32031ea29b,
                ])),
                Felt::new(BigInteger256([
                    0x40a8578501aa80d3,
                    0xebe5de0d9efa51ee,
                    0x2a9a06893d5e727c,
                    0x1018e07a54ab9b6e,
                ])),
                Felt::new(BigInteger256([
                    0xb0c8f478e9cf98bc,
                    0xfe25fd78e96c62be,
                    0x64eea16c0721821f,
                    0x1db2c1196ef7b6ac,
                ])),
                Felt::new(BigInteger256([
                    0x5e3d4b41212637d9,
                    0x4776d61eebe9cea4,
                    0xe8d8fafcfdc4d329,
                    0x2445c519faeb46a5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe05fdd4876bd976c,
                    0x165c9ae314ad04f1,
                    0x5d494158b95f8749,
                    0x0ec8049322c12292,
                ])),
                Felt::new(BigInteger256([
                    0x37673bf698d73d0f,
                    0x743b1f2c1f1361a3,
                    0x6e06867eca7b9163,
                    0x304f6e1ad0689360,
                ])),
                Felt::new(BigInteger256([
                    0x01da4e4227a4a615,
                    0xbddb965bb956e848,
                    0xd3cf3ef2ad92a0f7,
                    0x28b01b4caae2549b,
                ])),
                Felt::new(BigInteger256([
                    0xc36ef1d3fa71320c,
                    0xef5c50e5aa88fa93,
                    0xbca7e947aca51087,
                    0x22aecda1a4d5af08,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x902f4c39eeb2d318,
                    0xf84e5a64ec31b3cb,
                    0x36e707afd7879de8,
                    0x2397c4f8af3b7da2,
                ])),
                Felt::new(BigInteger256([
                    0xfe233a07b1a99c49,
                    0xc96c37be565ffdae,
                    0x5dc954c672e4ef96,
                    0x2f9f38ada3a401dc,
                ])),
                Felt::new(BigInteger256([
                    0x1a6b1a0b58128ab9,
                    0xe4df4f84636b2d63,
                    0x9988f6c961bfe475,
                    0x1ff366bbdc3f506e,
                ])),
                Felt::new(BigInteger256([
                    0x30346f7a6a7dc1d2,
                    0x2ce2b87863d9d551,
                    0x518e8a0f1d4f34d0,
                    0x2b052dc0ceb137f0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3b60f0167ed70295,
                    0x3601641ee4c26aae,
                    0x9329ce0f36477e5e,
                    0x1cd303b7d7e22472,
                ])),
                Felt::new(BigInteger256([
                    0x14ea7572264318ab,
                    0xabb44d79acec5a28,
                    0x0fbeb41eabe72194,
                    0x1cb99d9aa8854413,
                ])),
                Felt::new(BigInteger256([
                    0x29e1b54ba34cb8cb,
                    0xba40dbc24ceab335,
                    0x4a7fb8aa29076db7,
                    0x1f060fc86c1c4288,
                ])),
                Felt::new(BigInteger256([
                    0xec3b9ee86767e585,
                    0xfea3a75ca9619a01,
                    0xe8fe1e71a9849c6f,
                    0x114cc5407d9a828c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x81d7224b8031597b,
                    0x08b33f6e344f5a03,
                    0x60d22e526088974b,
                    0x1ee6359eb232aa0b,
                ])),
                Felt::new(BigInteger256([
                    0xa9718ebf81daec33,
                    0x41a332a065ae65fe,
                    0x959358999e183986,
                    0x2238e6586b654ce0,
                ])),
                Felt::new(BigInteger256([
                    0x833936fc9ef4be0e,
                    0x0c83950823b9737c,
                    0x776a9cb9a4a5307a,
                    0x286ec345654ba160,
                ])),
                Felt::new(BigInteger256([
                    0x074f95414cb5f05b,
                    0xe193ce6eb1887cb6,
                    0x51305c7be11e1b45,
                    0x086fb09375adb7a9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe23f91372b3c3bc7,
                    0xa73ee1a8906541cf,
                    0x9839476c9b64954c,
                    0x0349ec28f7bdc9a1,
                ])),
                Felt::new(BigInteger256([
                    0x05f7eaf2adb4a07a,
                    0xe89fa98b787c90bd,
                    0xf5970fa275f17496,
                    0x00880a64cb3796b7,
                ])),
                Felt::new(BigInteger256([
                    0xcc9c746417cb39b0,
                    0x267fe5d5f967b101,
                    0xb1a66216d51e5220,
                    0x127d23b9ae932aa6,
                ])),
                Felt::new(BigInteger256([
                    0x6ae0558c2738640d,
                    0x8155809c29b7537d,
                    0x818edb248837f5b8,
                    0x0cb3e587dbdbc2f5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x87a5a47cda595475,
                    0xac2ad125e0a414fe,
                    0x900411e86b22e227,
                    0x13f8d568b0bf1a1e,
                ])),
                Felt::new(BigInteger256([
                    0x2fdd3e02e30a23f9,
                    0x4ecc63561f20e5db,
                    0xab89679409cb4699,
                    0x0f368fd0bd16d55b,
                ])),
                Felt::new(BigInteger256([
                    0xf5f22e64d2730963,
                    0xd5d07a2e75a58a0e,
                    0x2981675ef3d1e8fc,
                    0x16ddd5f3f8db1de3,
                ])),
                Felt::new(BigInteger256([
                    0xba17ea536df0edbb,
                    0xdb1cb0fbe092ec44,
                    0x19644fac2428c4ef,
                    0x30254af22f293fc7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x84520942d4d17194,
                    0xbbc8381cac6fe524,
                    0x996a00c4d14e3ec4,
                    0x051a5948f07b2832,
                ])),
                Felt::new(BigInteger256([
                    0xd5fbe4b288cb7b94,
                    0x650a3a78b76f1a10,
                    0x05c8384ce8727c7e,
                    0x172ec5d97049fc2d,
                ])),
                Felt::new(BigInteger256([
                    0x26b276b9c918d28e,
                    0x016e4a1991a24aa9,
                    0x15c812731c4e2444,
                    0x172f6183c902be6d,
                ])),
                Felt::new(BigInteger256([
                    0xccfd7f2fb3f0e4c1,
                    0x5e39998c79629e9b,
                    0x5ba9a0208b1f892f,
                    0x1b891db748684cf2,
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
                Felt::new(BigInteger256([
                    0x0831d9741c148ac9,
                    0xa6cba06f4bdc9b6b,
                    0xc248aae6eb778615,
                    0x0f7803bb9d368ff2,
                ])),
                Felt::new(BigInteger256([
                    0x9b8661a6e92c9f6d,
                    0xd630218ce3793fd0,
                    0x0d617edbcb0585a9,
                    0x1285030635b2df73,
                ])),
                Felt::new(BigInteger256([
                    0xe9a3cc31cb1a37a6,
                    0x6e7617410f977990,
                    0x478555de7514433f,
                    0x2327aa0d885445b1,
                ])),
                Felt::new(BigInteger256([
                    0x109cf3b443b9443b,
                    0x7f9d0ed94e886c12,
                    0x8a607e4edf80bf6b,
                    0x0bb092e6b63676ec,
                ])),
                Felt::new(BigInteger256([
                    0x629b96f3c7f51897,
                    0x7949000c1c8b707f,
                    0xb6bd3d2274286cb7,
                    0x02591b286edf58e0,
                ])),
                Felt::new(BigInteger256([
                    0x3d2e3834d266405f,
                    0x8b295858a987d2ae,
                    0x2ede8dcc1d627a1f,
                    0x0c7d0b3dbcbc8ff1,
                ])),
                Felt::new(BigInteger256([
                    0xe241482106d1e488,
                    0xa945a06dccd3197c,
                    0x1dae958c75a113c5,
                    0x1f6ad1746b297ac3,
                ])),
                Felt::new(BigInteger256([
                    0xb05796ba5b97546d,
                    0xd2c5cccee537bcea,
                    0xf7a6372610a35d58,
                    0x16cf2f81fd7bf401,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2863771281a17ece,
                    0x7c59f719a062feb8,
                    0xbcee5d999f763af1,
                    0x23d5629f37511162,
                ])),
                Felt::new(BigInteger256([
                    0xacf9e2e9ac76fa93,
                    0x1b020926f0dfda58,
                    0x1443475c411f1a11,
                    0x0cb19891e00fe845,
                ])),
                Felt::new(BigInteger256([
                    0xaf1de9251df49ca7,
                    0x92edbe3266d9cb1d,
                    0xbbec04f6b3950c1b,
                    0x0c98da93cb0c90c8,
                ])),
                Felt::new(BigInteger256([
                    0xd5d554c0db5f6fc2,
                    0xfdc157767b1529d9,
                    0x029fee415f3c648f,
                    0x0820dbab0c216263,
                ])),
                Felt::new(BigInteger256([
                    0xc22e6667df370e28,
                    0x5bcd58f301018ee1,
                    0xf6369df14e9e8d5a,
                    0x1841634eba62d9a1,
                ])),
                Felt::new(BigInteger256([
                    0x7e3e5872ed311bf7,
                    0xdeed5f671dea9fe9,
                    0x42bcaa73878f3d44,
                    0x266b32ae024b523d,
                ])),
                Felt::new(BigInteger256([
                    0x46a6d43a8df8cefe,
                    0x34a290980b2e8ef0,
                    0x6b8d51ea85b4f2f6,
                    0x08b09294abf7b4cb,
                ])),
                Felt::new(BigInteger256([
                    0x9ded068aee465996,
                    0x203f7bdee5b63957,
                    0x24ad0b3a6720936c,
                    0x1065481e702cf6a4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5786889c7107c106,
                    0xb388ed49feaf41ac,
                    0xfe7ddede907c2879,
                    0x250aadb7fc0ef529,
                ])),
                Felt::new(BigInteger256([
                    0xb7316f44f3adf83a,
                    0xd08a289b7b14c563,
                    0x193d6f3c9f81a2f2,
                    0x0c0b57e7ca82c7d5,
                ])),
                Felt::new(BigInteger256([
                    0x4e4cfec62382395e,
                    0x1b17ff1672f9dba9,
                    0x5123db996232b82f,
                    0x04e6e56394996b41,
                ])),
                Felt::new(BigInteger256([
                    0x4c4d6865066a0179,
                    0xc83958134711c045,
                    0x54dd1991a3705377,
                    0x1db8fd2e3c722c86,
                ])),
                Felt::new(BigInteger256([
                    0xcc14176a79eb9756,
                    0x75f16689be3ca6f6,
                    0x001c1d611f451ccf,
                    0x221a7db1b74d34c2,
                ])),
                Felt::new(BigInteger256([
                    0xa98b513eb320c4b6,
                    0x55f526cf5ae9b45a,
                    0x8878316ab487cc7b,
                    0x20dde430eb470bd8,
                ])),
                Felt::new(BigInteger256([
                    0xed522f295db58727,
                    0xc2dfa570ccf3ce54,
                    0xf7ee0b435a5ff2ef,
                    0x1c233e2490026d40,
                ])),
                Felt::new(BigInteger256([
                    0x5f7d9a62ad1addd2,
                    0x0893fbc03fab5f36,
                    0xf4301479af7cf4a1,
                    0x1717340750c811a4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x86b7ceaafa1d05cc,
                    0x57c15265743b3f1d,
                    0x9dfac6560ddf2ec6,
                    0x03a65bd32a903b5f,
                ])),
                Felt::new(BigInteger256([
                    0x670e496a44f1d3fa,
                    0x93b687ba772fc136,
                    0x665c623925a7a11f,
                    0x28844389b8e39c60,
                ])),
                Felt::new(BigInteger256([
                    0x2d2dee0d635cf64c,
                    0xd97621396e012d18,
                    0xe71fe9ea0117ea37,
                    0x233e95be9c314045,
                ])),
                Felt::new(BigInteger256([
                    0x1c73ab98b8acdc12,
                    0x3be02eba95924157,
                    0x820e5ed6dd5d2f66,
                    0x06b6bce113284445,
                ])),
                Felt::new(BigInteger256([
                    0xe0f8fd6bc032c870,
                    0x342c6a2c185a3ac9,
                    0x3e1371e7d9c5a2de,
                    0x233c49d964739486,
                ])),
                Felt::new(BigInteger256([
                    0x1ae9b56693dffb4b,
                    0x889e9adce5ba77c0,
                    0x6486936ed3620606,
                    0x0218b5f22af807be,
                ])),
                Felt::new(BigInteger256([
                    0x8695dfc2a666e4c2,
                    0x9033625bc683bf42,
                    0xe9e1dd8510ea3202,
                    0x1b5a735e47425253,
                ])),
                Felt::new(BigInteger256([
                    0xa506f275a94c9cc3,
                    0xeac19a3cf69c68c5,
                    0xc9a61a4cb4373aed,
                    0x0a39d62a0d2eab06,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa5485c19792b794f,
                    0x5022c609af9686be,
                    0x79c6a1ddd3f5824d,
                    0x25bf64047b4a8e4f,
                ])),
                Felt::new(BigInteger256([
                    0x2836303dc249a83c,
                    0x27b7a4e5c67e1d27,
                    0xa8a1f48975db7a38,
                    0x2489378e26e39689,
                ])),
                Felt::new(BigInteger256([
                    0xa2621fada23abec2,
                    0xef6b6b5e5c7b2dd5,
                    0x8c0442a5ddf3af9c,
                    0x1fb1dc94f23f650f,
                ])),
                Felt::new(BigInteger256([
                    0x678668692635882f,
                    0xcf141caaf7528426,
                    0x97a983b8fce6d645,
                    0x119a8e6e621f4609,
                ])),
                Felt::new(BigInteger256([
                    0xec1fa17245133789,
                    0x8d479a93cf711838,
                    0x727d606e6bac7fca,
                    0x097e014854fda3a7,
                ])),
                Felt::new(BigInteger256([
                    0x5cf0d455fa439e27,
                    0xedd764eaa2bb5bf4,
                    0x42707e6c320f3595,
                    0x1b4927ea6a13b7be,
                ])),
                Felt::new(BigInteger256([
                    0x6d84f68b45bbb472,
                    0xdaadd3289ed1d931,
                    0x0043830bb5253d2b,
                    0x2312d0a4d449e938,
                ])),
                Felt::new(BigInteger256([
                    0x8717af9daae9550b,
                    0x5adb2e4030054323,
                    0xd4792f62c2290f04,
                    0x140bdf3bfa3c5cbf,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x32373932b206adf9,
                    0x5a0aefaa3ac6f497,
                    0x635e817a2d125d40,
                    0x0024d259db3a2989,
                ])),
                Felt::new(BigInteger256([
                    0x8a53b0c6c4e056ae,
                    0xc08c97c0f5e42265,
                    0x8531a769f740872b,
                    0x1b92c4946a838fef,
                ])),
                Felt::new(BigInteger256([
                    0xe07c79401429d0c9,
                    0x6fe2570ccd6ebbec,
                    0x95cf0d5d9167e852,
                    0x1d3d912812c00681,
                ])),
                Felt::new(BigInteger256([
                    0xe32226a1306b6724,
                    0x9a814743bb84d7e2,
                    0x1adfef1a5db22b0e,
                    0x2a0f88dbd98387c3,
                ])),
                Felt::new(BigInteger256([
                    0x226ad5080b9248a5,
                    0x3696374253c3ba9e,
                    0x9dcbad1105c9b85c,
                    0x0fd6488f1851c388,
                ])),
                Felt::new(BigInteger256([
                    0x3a45c1314c1222e7,
                    0x79d4f02b859bba62,
                    0x08607f5603ad4f5e,
                    0x119eb4e863002789,
                ])),
                Felt::new(BigInteger256([
                    0xc37af46ac9a7eb31,
                    0x643a229eea458c9d,
                    0x88aed63f257078b6,
                    0x08553d0e899b61f8,
                ])),
                Felt::new(BigInteger256([
                    0x3fba5e8b0cf98312,
                    0xc14db9ca0e169f4d,
                    0xfdbbcbb938ec21f4,
                    0x003eb38ac82780e9,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xea2d7b1f2ce58b77,
                    0xc960f3f1f4729221,
                    0x0a31399bf35d1718,
                    0x2e328b4f4778a7b3,
                ])),
                Felt::new(BigInteger256([
                    0x990889ff3952435c,
                    0xbeb15a2a05c3d1e4,
                    0xd65ec27602f5bf17,
                    0x178fe02a82ffc4ad,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7eefbfbedb9cc6ec,
                    0x0cd5ffcd1ece5a55,
                    0x298d8120c39ec579,
                    0x2755b4a22d1f9341,
                ])),
                Felt::new(BigInteger256([
                    0xf449d25a5365e5d7,
                    0x962670af76760a82,
                    0x229ddbf4a097ea3e,
                    0x0e60d4a5ca8aec27,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x876dc5b38059fa08,
                    0xaf99e8f5d8114e2c,
                    0xca92b0404a15be6c,
                    0x2a1a3d4b72165947,
                ])),
                Felt::new(BigInteger256([
                    0x62c516af4a53bb65,
                    0x9bdb499b22725605,
                    0x5b22bbcfb9a1ed48,
                    0x03fa57216e6541ea,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa6199f73c5e5403a,
                    0x3cb6c6ad659222ac,
                    0x78c83a94e570cfe3,
                    0x0713d16cec71d704,
                ])),
                Felt::new(BigInteger256([
                    0xbeb5a1b3bacb71d4,
                    0xcc160580612a91a9,
                    0x725e2a0ff59f498d,
                    0x2299ed49940ca23f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa6610b8b8ff1b894,
                    0xd4de00779562bd43,
                    0xb60214020535a6bc,
                    0x097a07be0f44b7a9,
                ])),
                Felt::new(BigInteger256([
                    0x1fe1e8fa7e1f5fff,
                    0x10264d459544436b,
                    0x26bb9efaa70a5f44,
                    0x29769a8ec9f1acf8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd585d996aef463d6,
                    0x507384c07f37ff72,
                    0xf8fae3fe6231910b,
                    0x27912638ae1bb63f,
                ])),
                Felt::new(BigInteger256([
                    0x76fb744dc89c0cca,
                    0x59d042f6aa049575,
                    0x029fd24db278c61e,
                    0x14ded40a31a00ab2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb37cd1438c4c7943,
                    0x88de524e89b3eaff,
                    0x51f278ceaa1bfcd6,
                    0x12e9f877f9d93b7c,
                ])),
                Felt::new(BigInteger256([
                    0x279bbf4ed6a127ca,
                    0xbcaa29554618b2ff,
                    0xf00e812c2eb6cdd5,
                    0x0092ada3f9406d52,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x439773b76edf5812,
                    0x9f368f8c999a672f,
                    0x9e2089862238a3f6,
                    0x20b52387d15eb3d5,
                ])),
                Felt::new(BigInteger256([
                    0x596b1e73f73fa0ae,
                    0x307182436d5e4a29,
                    0x17e8b4489b4eceab,
                    0x20ee31cd1f2716eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfd04d612fa4ff216,
                    0xa5d43456a8217b6d,
                    0x9115cbccc267fdc7,
                    0x05551249975e6711,
                ])),
                Felt::new(BigInteger256([
                    0x75db185ae47003a9,
                    0x1dc0a2d236bcd8cb,
                    0xec2ed16c9d136194,
                    0x07db71db2952b2cd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7906f2f8ba8dc5e8,
                    0x1a22090b4d145744,
                    0x490e72f24f8bca6c,
                    0x0ff3574e5556d655,
                ])),
                Felt::new(BigInteger256([
                    0x7a63d7eb0b53c4a3,
                    0xcccdcddcdaf6d07b,
                    0x1db8adce481bec25,
                    0x23e84e8823e3f9d0,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
