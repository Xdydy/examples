#include <iostream>

#define FUNC(name) void lygg##name() {std::cout << "lygg " << #name << std::endl;}

FUNC(sb)
FUNC(ysmg)

int main() {
    lyggsb();
    lyggysmg();
}