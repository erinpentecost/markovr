extern crate markovr;

pub fn main() {
    // Create a new, second-order Markov Chain for generating month names.
    let mut m = markovr::MarkovChain::<char>::new(2, &[0, 1]);

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
thout
paopi
hathor
emshir
paremhat
paremoude
pashons
paoni
epip
mesori
brumaire
frimaire
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
        // Handle first characters.
        // $ is used as a beginning-of-word indicator.
        // % is also needed for that indicator since
        // the Markov chain is second-order.
        m.train(&['$', '%'], row[0], 1);
        m.train(&['%', row[0]], row[1], 1);
        // Handle middle characters
        for c in 2..(row.len() - 1) {
            m.train(&[row[c - 2], row[c - 1]], train[r][c], 1);
        }
        // Cap the end.
        m.train(&[row[row.len() - 2], row[row.len() - 1]], ' ', 1);
    }

    // Generate values from the model.
    for _ in 0..12 {
        'gen: loop {
            let mut month = Vec::new();
            let mut lastlast = Some('%');
            let mut last: Option<char> = m.generate_from_partial(&[Some('$'), Some('%')]);
            while let Some(c) = last {
                // Did we end?
                if c == ' ' {
                    break;
                } else {
                    // Tack on the character
                    month.push(c);
                    // Find the next character
                    let window: [char; 2] = [lastlast.unwrap(), last.unwrap()];
                    lastlast = last;
                    last = m.generate(&window);
                }
            }
            let month_name: String = month.into_iter().collect();
            print!("{}\n", month_name);
            break 'gen;
        }
    }
}
