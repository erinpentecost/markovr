extern crate markovr;

pub fn main() {
    // Create a new, fourth-order Markov Chain.
    // We'll keep track of each orthogonal neighbor,
    // and allow for any one of them to be unknown.
    let mut m = markovr::MarkovChain::<char>::new(4, &[0, 1, 2, 3]);

    let train: Vec<Vec<char>> = "           
 ┏━━━┓     
 ┃   ┃     
 ┃   ┣━━━┓ 
 ┃   ┃   ┃ 
 ┃   ┣━━━┛ 
 ┃   ┃     
 ┗━━━┛     
           
"
    .lines()
    .map(|c| c.chars().take(12).collect())
    .collect();

    // Train the model.
    for r in 1..(train.len() - 1) {
        let ref row = train[r];
        for c in 1..(row.len() - 1) {
            // Build up a view of the neighbors.
            let neighbors = &[
                train[r - 1][c],
                train[r][c - 1],
                train[r][c + 1],
                train[r + 1][c],
            ];
            m.train(neighbors, train[r][c], 1);
        }
    }

    // Generate values from the model.
    const DIM: usize = 16;
    let mut map: [[Option<char>; DIM]; DIM];
    'gen: loop {
        map = [[None; DIM]; DIM];
        // Fill in spaces around the border. This isn't necessary,
        // but should prevent dangling lines in the output.
        for i in 0..DIM {
            map[i][0] = Some(' ');
            map[i][DIM - 1] = Some(' ');
            map[0][i] = Some(' ');
            map[DIM - 1][i] = Some(' ');
        }
        // Iterate on all non-None spaces and fill them in.
        for r in 1..(DIM - 1) {
            for c in 1..(DIM - 1) {
                let neighbors = &[map[r - 1][c], map[r][c - 1], map[r][c + 1], map[r + 1][c]];

                map[r][c] = m.generate_from_partial(neighbors);
                match map[r][c] {
                    Some(_) => {}
                    // We saw a case that wasn't in our training data,
                    // so throw it away and try again.
                    None => {
                        continue 'gen;
                    }
                }
            }
        }
        break 'gen;
    }

    for r in 1..(DIM - 1) {
        for c in 1..(DIM - 1) {
            match map[r][c] {
                Some(v) => print!("{}", v),
                None => print!("?"),
            }
        }
        print!("\n");
    }
    // Prints:
    /*
           ┏━━━━┓
           ┃    ┃
           ┃    ┃
      ┏━━━━┛    ┃
      ┃         ┃
    ┏━┛         ┃
    ┃           ┃
    ┃           ┃
    ┃           ┃
    ┣━━━━━━━━━━━┛
    ┃
    ┣━━━┓
    ┃   ┃
    ┗━━━┛
       */
}
