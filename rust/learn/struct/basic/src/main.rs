struct User {
    active:bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

fn main() {
    let mut user1 = User {
        email: String::from("EMAIL"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };
    user1.sign_in_count = 2;

    println!("{}", user1.sign_in_count);

    let user2 = User {
        email: String::from("EMAIL2"),
        username: String::from("someusername1234"),
       ..user1
    };
    println!("{}", user2.email);
    println!("{}", user2.username);
    println!("{}", user2.active);
}