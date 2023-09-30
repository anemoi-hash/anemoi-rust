//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiEdOnBls12_377_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiEdOnBls12_377_4_3 {
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
            state[i] += Felt::from_le_bytes_mod_order(&buf[..]);
            i += 1;
            if i % RATE_WIDTH == 0 {
                AnemoiEdOnBls12_377_4_3::permutation(&mut state);
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
            AnemoiEdOnBls12_377_4_3::permutation(&mut state);
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
                AnemoiEdOnBls12_377_4_3::permutation(&mut state);
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
            AnemoiEdOnBls12_377_4_3::permutation(&mut state);
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
        AnemoiEdOnBls12_377_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiEdOnBls12_377_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiEdOnBls12_377_4_3::permutation(&mut state);

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

        let mut state = elems.to_vec();
        AnemoiEdOnBls12_377_4_3::permutation(&mut state);

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

    use super::super::MontFp;
    use super::*;
    use ark_ff::BigInteger;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![MontFp!(
                "3245888732545938703293848064370856564777841805765396792997613448749116772753"
            )],
            vec![
                MontFp!(
                    "7663394338249865628842769414268966392219678610153970312684440283159581613217"
                ),
                MontFp!(
                    "6271904162410896908729341222426710964855145842094919193581967140936933284923"
                ),
            ],
            vec![
                MontFp!(
                    "7969334835959175341708732577328788201615587299246107167659857871324493208835"
                ),
                MontFp!(
                    "7493961423969253751307419581028383478098016292045659784917713206986998549296"
                ),
                MontFp!(
                    "4026170068082212382064346783732235106988068470245368593324116477525361418264"
                ),
            ],
            vec![
                MontFp!(
                    "2666117871319292549109278078687092690881141726197129135322956865265002280242"
                ),
                MontFp!(
                    "6509697908608382218025477966055018129401065002540626665823304793770577635119"
                ),
                MontFp!(
                    "1271169338579653709528628099295006826522129092665839774514071738347273016900"
                ),
                MontFp!(
                    "6227043312022748939281805745739526186202679329496983601547709929059252900150"
                ),
            ],
            vec![
                MontFp!(
                    "2909735215509804401941047549197284613681174544937466475970835246477891033436"
                ),
                MontFp!(
                    "5599520051368633913843848108157638779058707144916128517158704913308108424747"
                ),
                MontFp!(
                    "3432680043190079867020927346206445943703719322355884967763212834677264837833"
                ),
                MontFp!(
                    "947456959501802677678449235943191631201751906080558756492421326409270025507"
                ),
                MontFp!(
                    "770101993108319020437386465470482367547039566195436746871113992043063945007"
                ),
            ],
            vec![
                MontFp!(
                    "2080443911568684752411149575349419122512153854110376483525925460103994871352"
                ),
                MontFp!(
                    "6201195192392064815455731467660762987281772282324903591758021941563073326726"
                ),
                MontFp!(
                    "5816509240437701961737463357848466910945009150646501054530732013884280407532"
                ),
                MontFp!(
                    "364898586858423812020984816241377942240805600237616206986914820203073961940"
                ),
                MontFp!(
                    "7545609520856506738146634501223899964449171533615289497205422583748185923883"
                ),
                MontFp!(
                    "3267788309670351784476251074531266658544590980628771747836642182398089645088"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "2826852558969682283160965063399396013408991576304937928545694637519109987876"
            )],
            [MontFp!(
                "715813639245542616319007380674059374635588794208688983677833596359619725647"
            )],
            [MontFp!(
                "876091172931230864992957830321534820967228680519015826743030886193379751284"
            )],
            [MontFp!(
                "7832641788849582971659980460374408607868841084869548467467350585852449552347"
            )],
            [MontFp!(
                "1769025083902686531720248316128771525661472288104261222432854562461627706893"
            )],
            [MontFp!(
                "3508965325475653084247845767237725060146495017552397552195176578350789058908"
            )],
            [MontFp!(
                "1878818611437899959920600011342139010838536479547266920323586032506688840792"
            )],
            [MontFp!(
                "7419422835010040190368651561187392128505524499673594534865952878287607357198"
            )],
            [MontFp!(
                "3056644334375883027446266166030682099237729268530996862502216213378803940298"
            )],
            [MontFp!(
                "618929239659418549046582686223990802620595245882301785085280228605376233832"
            )],
        ];

        for (index, (input, expected)) in input_data.iter().zip(output_data).enumerate() {
            println!("{:?}", index);
            assert_eq!(
                expected,
                AnemoiEdOnBls12_377_4_3::hash_field(input).to_elements()
            );
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "2826852558969682283160965063399396013408991576304937928545694637519109987876"
            )],
            [MontFp!(
                "715813639245542616319007380674059374635588794208688983677833596359619725647"
            )],
            [MontFp!(
                "876091172931230864992957830321534820967228680519015826743030886193379751284"
            )],
            [MontFp!(
                "7832641788849582971659980460374408607868841084869548467467350585852449552347"
            )],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 124];
            bytes[0..31].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..31]);
            bytes[31..62].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..31]);
            bytes[62..93].copy_from_slice(&input[2].into_bigint().to_bytes_le()[0..31]);
            bytes[93..124].copy_from_slice(&input[3].into_bigint().to_bytes_le()[0..31]);
            assert_eq!(
                expected,
                AnemoiEdOnBls12_377_4_3::hash(&bytes).to_elements()
            );
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [
                MontFp!(
                    "4166825787354029787334922406033290250172572810107272783449272467370104361499"
                ),
                MontFp!(
                    "443119219083466133166448152339477273242820198154691037927464243515191538965"
                ),
            ],
            [
                MontFp!(
                    "546351612581094746909612373291868483487237596382940402109210862670055071860"
                ),
                MontFp!(
                    "4683692898576499548827723194744461803813133049085680138685198838036731652343"
                ),
            ],
            [
                MontFp!(
                    "3239548400000290705362327910139849710833184455393184789726734156259593056824"
                ),
                MontFp!(
                    "429426852404684922799838349190498511339808948377701253268888394251603161550"
                ),
            ],
            [
                MontFp!(
                    "2040858442524663883803034136807049307596283283862686075056479988745491268025"
                ),
                MontFp!(
                    "8233028930313417211499327277472739989778672106814084714224282121099996419938"
                ),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiEdOnBls12_377_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected.to_vec(),
                AnemoiEdOnBls12_377_4_3::compress_k(input, 2)
            );
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "4609945006437495920501370558372767523415393008261963821376736710885295900464"
            )],
            [MontFp!(
                "5230044511157594295737335568036330287300370645468620540794409700706786724203"
            )],
            [MontFp!(
                "3668975252404975628162166259330348222172993403770886042995622550511196218374"
            )],
            [MontFp!(
                "1829425623409710671053536475498242765999056055522706961345528653928078448922"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected.to_vec(),
                AnemoiEdOnBls12_377_4_3::compress_k(input, 4)
            );
        }
    }
}
