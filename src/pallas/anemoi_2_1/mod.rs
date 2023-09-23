//! Implementation of the Anemoi permutation

use super::{sbox, Felt, MontFp};
use crate::{Anemoi, Jive, Sponge};
use ark_ff::{One, Zero};
/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;

// ANEMOI CONSTANTS
// ================================================================================================

/// Function state is set to 2 field elements or 64 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 21 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 21;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over Pallas basefield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiPallas_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiPallas_2_1 {
    const NUM_COLUMNS: usize = NUM_COLUMNS;
    const NUM_ROUNDS: usize = NUM_HASH_ROUNDS;

    const WIDTH: usize = STATE_WIDTH;
    const RATE: usize = RATE_WIDTH;
    const OUTPUT_SIZE: usize = DIGEST_SIZE;

    const ARK_C: &'a [Felt] = &round_constants::C;
    const ARK_D: &'a [Felt] = &round_constants::D;

    const GROUP_GENERATOR: u32 = sbox::BETA;

    const ALPHA: u32 = sbox::ALPHA;
    const INV_ALPHA: Felt = sbox::INV_ALPHA;
    const BETA: u32 = sbox::BETA;
    const DELTA: Felt = sbox::DELTA;

    fn exp_by_inv_alpha(x: Felt) -> Felt {
        sbox::exp_by_inv_alpha(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                MontFp!(
                    "14427893260455109705156664042356718870640358762168844127801340015397735141384"
                ),
                MontFp!(
                    "13138696979498515255251127567565619801757554887203558360380530676670571486013"
                ),
            ],
            [
                MontFp!(
                    "5982785702201799305786867067988158272995997210813762482456562383325396457307"
                ),
                MontFp!(
                    "11157578518762194736934882103054173468937876478529400613348074898361123904134"
                ),
            ],
            [
                MontFp!(
                    "28438295326901256591313961101106709090488938900652514646147770410889552753021"
                ),
                MontFp!(
                    "21984098443870633452751627308924492469490828327318452161617021106419179787063"
                ),
            ],
            [
                MontFp!(
                    "1478473684176282421545569025094169832604223179056238532794256740200864105805"
                ),
                MontFp!(
                    "10434311809031514623895595215935435926737673069788553325979295545028061815206"
                ),
            ],
            [
                MontFp!(
                    "14781888674079440730534661652152630801376143896208339356492088260293981316061"
                ),
                MontFp!(
                    "5946100447321140538927082795254280680319672054848714933905562692357791209656"
                ),
            ],
            [
                MontFp!(
                    "21890634181453312985299942362110105598465044840040594871483168981564503243109"
                ),
                MontFp!(
                    "13399999702513795139595738682345741616628808395869269194943347354713043743547"
                ),
            ],
        ];

        let output = [
            [
                MontFp!(
                    "11579208923731619542357098500868790785345222592776624286381870705739987052135"
                ),
                Felt::zero(),
            ],
            [
                MontFp!(
                    "21735578927475698800610569875486878598709331368954848907691439386750294515554"
                ),
                MontFp!(
                    "14915059756306458668798776150463074115887270782104658469523764643553127876149"
                ),
            ],
            [
                MontFp!(
                    "8778638346924233418081111828888910239654127576710317834940372077697612057229"
                ),
                MontFp!(
                    "22051619713425230766531768624512758113922728761788164591626179149121958488460"
                ),
            ],
            [
                MontFp!(
                    "11579208923731619542357098500868790785345222592776624286381870705739987052141"
                ),
                MontFp!(
                    "28948022309329048855892746252171976963363056481941560715954676764349967630336"
                ),
            ],
            [
                MontFp!(
                    "15057666972438544655529092362515483735243961223136603604240845793436024282167"
                ),
                MontFp!(
                    "18426015703030010997579367729105852162907463595338775756615551204155238010747"
                ),
            ],
            [
                MontFp!(
                    "12784112979346968338857027566852413713742326351825438806654554069180360520239"
                ),
                MontFp!(
                    "19489047326280876009902092412625999917701345308237436085355219040385444324475"
                ),
            ],
            [
                MontFp!(
                    "1074062906705051625340891993854330434113074614285707814128597733756511897673"
                ),
                MontFp!(
                    "26232282506455021707774466392106533529531735952259475413117013981021691187782"
                ),
            ],
            [
                MontFp!(
                    "28595818858315685708975605358852866393238263106947165825818134308983563312207"
                ),
                MontFp!(
                    "14642156524169052991136322603510653318103271560700097395676529642867165495210"
                ),
            ],
            [
                MontFp!(
                    "11754355108941335665457273660950498145132042536979472128903765695489200586844"
                ),
                MontFp!(
                    "15784254972170032700858756124355809739660845922750435361027936389430659344563"
                ),
            ],
            [
                MontFp!(
                    "15232629547649820334265900608014588244555627611748155043487778827952582329153"
                ),
                MontFp!(
                    "15198336594802842939444122386359795234810128717205332049155326138169684053519"
                ),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiPallas_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
