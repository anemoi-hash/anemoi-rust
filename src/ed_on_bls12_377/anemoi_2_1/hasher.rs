//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiEdOnBls12_377_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiEdOnBls12_377_2_1 {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 32];
        for (i, chunk) in bytes.chunks(31).enumerate() {
            if i < num_elements - 1 {
                buf[0..31].copy_from_slice(chunk);
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
            state[0] += Felt::from_le_bytes_mod_order(&buf[..]);
            AnemoiEdOnBls12_377_2_1::permutation(&mut state);
        }
        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        for &element in elems.iter() {
            state[0] += element;
            AnemoiEdOnBls12_377_2_1::permutation(&mut state);
        }

        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // We use internally the Jive compression method, as compressing the digests
        // through the Sponge construction would require two internal permutation calls.
        let result = Self::compress(&Self::Digest::digests_to_elements(digests));
        Self::Digest::new(result.try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiEdOnBls12_377_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiEdOnBls12_377_2_1::permutation(&mut state);

        vec![state[0] + state[1] + elems[0] + elems[1]]
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

    use super::super::MontFp;
    use super::*;
    use ark_ff::BigInteger;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![MontFp!(
                "2701794724129673170044881656924003678119508097167887646056013505081505153941"
            )],
            vec![
                MontFp!(
                    "6136112418425776110987395661408630531947256723654514515474099325642827368195"
                ),
                MontFp!(
                    "1750665747509629654143681886817319364490061620706939457796280733677342840536"
                ),
            ],
            vec![
                MontFp!(
                    "7419302752507955507727625165019065254163119937378581731857420479322421089499"
                ),
                MontFp!(
                    "4559650952616111375934279109757813936885978902236988815039265000756509380033"
                ),
                MontFp!(
                    "1135233996092168754628300471243482957747287551785928066349278596976141683924"
                ),
            ],
            vec![
                MontFp!(
                    "1043102290181650327389949602749036862032433278523840306868639917710387374771"
                ),
                MontFp!(
                    "4307491562116361618597497082323305958665690819685789478378073202108247466174"
                ),
                MontFp!(
                    "8276002717822051727248370396378451046543961863530952458242898618930392259498"
                ),
                MontFp!(
                    "697823225422979508435425451318633448160314726681841164485634653488707337155"
                ),
            ],
            vec![
                MontFp!(
                    "3511613069045600668201060103623350362861378594064425563036950657064452808148"
                ),
                MontFp!(
                    "164826560488580658427980384618027169790561646104634207479832305366799917896"
                ),
                MontFp!(
                    "7725367446600896112606560721239505513493098704961497582884899050064367743267"
                ),
                MontFp!(
                    "1304519449128211841092493704609228683609347511981155101682539749031135624359"
                ),
                MontFp!(
                    "2253160602893615828475245752494423392208457478787057653939826714992705372503"
                ),
            ],
            vec![
                MontFp!(
                    "6778459080375078354240530484738211312590141386355539256680627032395387751580"
                ),
                MontFp!(
                    "93624049550182582977412113403479422800060943063169915046698223121519200466"
                ),
                MontFp!(
                    "7722399227155929647209203618018966141115046905466472477162114078434889925803"
                ),
                MontFp!(
                    "4362457042764236508126106021709659576958225187859016903781545398127098366287"
                ),
                MontFp!(
                    "1635598988571107961875869710676973678440790488903729803300931707418622110543"
                ),
                MontFp!(
                    "4340733038522388446865752953560902505670573737797919431024824522904869142592"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "6038547479163037986409334235918516245718888275963429958045868475748310870262"
            )],
            [MontFp!(
                "5290322214010334967587282982485974931105631029110653070735751242693563614504"
            )],
            [MontFp!(
                "3724359895368550768973893174946038660878167910590705539560114603470279328816"
            )],
            [MontFp!(
                "5395258772068762925789170702532573198276721923542161779150941414089330072040"
            )],
            [MontFp!(
                "6437389331144526233027251753132997194151245817848750712560989451783332250690"
            )],
            [MontFp!(
                "6387014775150552799372860356098991172191065503357197351459808793371394774834"
            )],
            [MontFp!(
                "779756219490472826418676381497465819568380834103446740858202937881929583962"
            )],
            [MontFp!(
                "5281691314668098790968241205590675464346678528867299504089015084730289888849"
            )],
            [MontFp!(
                "620311156215507825863696040333743484531686107885002700624515445245202549984"
            )],
            [MontFp!(
                "6791364295911321742758975425730448919938309444628380874854471393642988927668"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiEdOnBls12_377_2_1::hash_field(input).to_elements()
            );
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "6038547479163037986409334235918516245718888275963429958045868475748310870262"
            )],
            [MontFp!(
                "5290322214010334967587282982485974931105631029110653070735751242693563614504"
            )],
            [MontFp!(
                "3724359895368550768973893174946038660878167910590705539560114603470279328816"
            )],
            [MontFp!(
                "5395258772068762925789170702532573198276721923542161779150941414089330072040"
            )],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..31]);
            bytes[31..62].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..31]);

            assert_eq!(
                expected,
                AnemoiEdOnBls12_377_2_1::hash(&bytes).to_elements()
            );
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "676553123262956770831412211816454043839638653372636331333841739259626467495"
            )],
            [MontFp!(
                "961399696453657341825013342908082065226592910566258118404273138500594033718"
            )],
            [MontFp!(
                "7483032027386392696108818818965861527781219423978929688858862524367943483025"
            )],
            [MontFp!(
                "1507470731653473592080957199708477983334364641992160367637834033297977648401"
            )],
        ];
        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiEdOnBls12_377_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected.to_vec(),
                AnemoiEdOnBls12_377_2_1::compress_k(input, 2)
            );
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiEdOnBls12_377_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
