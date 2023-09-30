//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiBls12_377_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiBls12_377_2_1 {
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
            state[0] += Felt::from_le_bytes_mod_order(&buf[..]);
            AnemoiBls12_377_2_1::permutation(&mut state);
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
            AnemoiBls12_377_2_1::permutation(&mut state);
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

impl Jive<Felt> for AnemoiBls12_377_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiBls12_377_2_1::permutation(&mut state);

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
                "65130260748368920281538885457409646425128704771983086943299167921824082721049285243519137611409340273689860222034"
            )],
            vec![
                MontFp!(
                    "73797961748931766908339625579651713018616922111622571415397160838048516757655188395762598569219304577396579940212"
                ),
                MontFp!(
                    "7861589786408085707689481693281128788600110174329923978447814285409189128620932703613937011392774494885985226525"
                ),
            ],
            vec![
                MontFp!(
                    "1140282778423381159981403229211891003512363678702887913360285870489175768152149912680788819405787688508580387013"
                ),
                MontFp!(
                    "141384838363351399873940301976002505195971622286352023127172314331498465909403199982938060958160534125198952518483"
                ),
                MontFp!(
                    "110803271563953149778921849883683629538799030785474817204877708917801945758414501935554082772341683220938118668067"
                ),
            ],
            vec![
                MontFp!(
                    "26616198569712800105002063589623699462693091616022340620295926954254126527593743290779102940053362318537901145618"
                ),
                MontFp!("104491651967893928999419237550673823280353613164216358101691523500055496068773447774727170095913322380468775279673"
                ),
                MontFp!("31762832281976421635912745263980035670338339868329759702710265895140072606279566055970021497932315937059456131084"
                ),
                MontFp!("61042929895265422945932246361382789301640724167411316131115412135495856186033324728637382555043049179076512145800"
                ),
            ],
            vec![
                MontFp!(
                    "168263812423116090850908266182670095907331550464733125028091142531643095263542772981934357491226981473568726303570"
                ),
                MontFp!("208385159484034876412537460471514125302714559668738705083276621893091142540197238563265681352635931666199770743187"
                ),
                MontFp!("220757014146280079766048715901297267813986997352385737068372383694786531762408106883129439658292326587531966427631"
                ),
                MontFp!("41090064652027785748072325900048859510298733047564998070048808467693285481990698992053932294383366853044162990119"
                ),
                MontFp!("125604351136808645401317087823155229452054431407496991291310654370467520104068424127084351952549617499131990810873"
                ),
            ],
            vec![
                MontFp!(
                    "247628606660679802832676757276755735193812163573634257307997292741188782358199048598718807748312696289179213274128"
                ),
                MontFp!("250285387017776522861524502950568160213922157357751325973260311460671993212512088663173990859834090693599020001585"
                ),
                MontFp!("186580800935244205460280972534709417910612425367599138183607620557878177273984559587148632379743110542814215461841"
                ),
                MontFp!("44084483241185906992216920550271710399005121743364549523957794723857726067739950381816301148798975267985079128778"
                ),
                MontFp!("156525723948122246198346812412151340420273372770240397269538777267321057240477045693659249540168654520526231498456"
                ),
                MontFp!("108312085723547448198286038654386134448772057335221514195062902445420331141188859478524432449455443961755150699176"),],];

        let output_data = [
[MontFp!("27766465197669188653483451021305744617555593440025590298723517537135767244889896997910428916633564726352291994040"),]
,[MontFp!("57263075151006122338205435543640935761456304939537717323318663152752487068744340342154701953805188173397092391849"),]
,[MontFp!("28656290678635440726024641420568187346949513468055817168026158674268757481074951900527253466865715002100958848572"),]
,[MontFp!("101516977039721717479267790618721568125366352300802139849671081548548238487469150270830726661010442516157652086420"),],[MontFp!("96718568646851196308764437402098844076353897856835340367996287290968703951475662134946961994095599456701980101992"),],[MontFp!("115082590565296901950005342709495703166532825943284993159676694709318986676003828515903933820139996724899780475467"),],[MontFp!("150686962604339286983097678752670162248533694992708214679286230499016401769545376453957127344127168623715356318480"),],[MontFp!("77267640359088653316497279592219939547515400085866299833766629089238393990847071854873228657573663886084842547568"),],[MontFp!("144977940836759972004452187533433746460227772841787147819270051023488387918933399121127770033162711863532233281550"),],[MontFp!("220934957760797032996098360910583068163014799284956613702229338213707633771680017466752797283124482866808160529468"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_377_2_1::hash_field(input).to_elements()
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
            [MontFp!("27766465197669188653483451021305744617555593440025590298723517537135767244889896997910428916633564726352291994040"),]
,[MontFp!("57263075151006122338205435543640935761456304939537717323318663152752487068744340342154701953805188173397092391849"),]
,[MontFp!("28656290678635440726024641420568187346949513468055817168026158674268757481074951900527253466865715002100958848572"),]
,[MontFp!("101516977039721717479267790618721568125366352300802139849671081548548238487469150270830726661010442516157652086420"),],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 94];
            bytes[0..47].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..47]);
            bytes[47..94].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..47]);

            assert_eq!(expected, AnemoiBls12_377_2_1::hash(&bytes).to_elements());
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
[MontFp!("144903599339269640325915928269460698358999629330281610104254835795032412421129696153413585003367453521301422819647"),],[MontFp!("190360647877866053699086757392267254158554044789267077910903482174071640043746188016588062154535637876714285586573"),],[MontFp!("86101712911018799954483658062361107666176814799901041368170314107402701836689532637345770262332171396043117097323"),],[MontFp!("242492377784223683397457947583707192700817123707687819516074636506685056725347958443191559361660571274001628586416"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_377_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_377_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
