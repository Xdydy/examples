fn main () {
    let mut s = String::from("hello");
    s.push_str( ", world!");
    println!("{s}");

    let s1 = String::from("hello");
    let s2 = s1;

    // println!("{s1}"); // error
    println!("{s2}");

    let s3 = String::from("hello");
    let s4 = s3.clone();
    println!("{s3}, {s4}");

    let x = 5;
    let y = x;
    println!("{x}, {y}");

    take_ownership(s);

    makes_copy(5);
}

fn take_ownership(str: String) {
    println!("{str}");
}

fn makes_copy(int: i32) {
    println!("{int}");
}