extern crate markovr;

pub fn main() {
    // Create a new, first-order Markov Chain for generating month names.
    let mut m = markovr::MarkovChain::<char>::new(1, &[0]);

    let train: Vec<Vec<char>> = "january
february
march
april
may
june
july
august
september
october
november
december
nisan
iyar
sivan
tammuz
av
elul
tishri
marcheshvan
kislev
tevet
shevat
adar
muharram
safar
rajab
shaban
ramadan
shawwal
caitra
vaikasi
jyestha
ashada
sravana
bhadrapada
asvina
kartika
maargazhi
pausa
magha
chet
vaisakh
jeth
harth
sawan
bhadon
assu
katak
maghar
poh
magh
phagun
gormanuour
ylir
morsugur
porri
goa
einmanuour
harpa
skerpla
solmanuour
heyannir
tvimanuour
haustmanuour
thout
paopi
hathor
koiak
tooba
emshir
paremhat
paremoude
pashons
paoni
epip
mesori
vendemiarie
brumaire
frimaire
nivose
pluviose
ventose
germinal
floreal
prairial
messidor
thermidor
fructidor
"
    .lines()
    .map(|c| c.chars().collect())
    .collect();

    // Train the model.
    for r in 0..(train.len() - 1) {
        let ref row = train[r];
        for c in 1..(row.len() - 1) {
            m.train(&[row[c - 1]], train[r][c], 1);
        }
        // Cap the end
        m.train(&[row[row.len()-1]], ' ', 1);
    }

    // Generate values from the model.
    'gen: loop {
        let mut month = Vec::new();
        let mut last: Option<char> = m.generate_from_partial(&[None]);
        while let Some(c) = last{
            // Did we end?
            if c == ' '{
                break;
            }
            else {
                // Tack on the character
                month.push(c);
                // Find the next character
                last = m.generate(&[last.unwrap()]);
            }
        }
        // Throw away items that are too short
        if month.len() < 4 {
            month.clear();
            continue 'gen;
        }
        let month_name: String = month.into_iter().collect();
        print!("{}\n", month_name);
        break 'gen;
    }
}