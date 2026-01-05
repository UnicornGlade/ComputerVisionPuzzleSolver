// Стиль кодирования опирается на LLVM code style, но:
// - 4 пробела на каждый отступ
// - 120 символов ширина строки
// - code style описан в /.clang-format
// Ниже приведен модифицированный пример взятый отсюда: https://github.com/motine/cppstylelineup

#include <iostream>

namespace western {

enum ShootingHand { LEFT, RIGHT };

class Cowboy {
public:
    Cowboy();
     Cowboy(int gun_count);
    ~Cowboy();

    void shoot(std::string who);
    int getAge() const { return age_; }

protected:
    void makeBang(const int &how_many);

    int gun_count_;
    int age_;
};

Cowboy::Cowboy() : age_(45), gun_count_(2) {
    std::cout << "I am alive!" << std::endl;
}

Cowboy::Cowboy(int gun_count) : age_(45), gun_count_(gun_count) {
    // And this is the longest inline comment you have every seen. Lorem Ipsum. Bacon! Cheese. Bread. Beer. Fish.
    // Mobile fridge. More random words come here...
    std::cout << "Howdy!" << std::endl;
}

Cowboy::~Cowboy() {
    std::cout << "RIP" << std::endl;
}

void Cowboy::shoot(std::string who) {
    age_++;
    int some_number = 5 + 7 * 3;
    makeBang(some_number);
    if (some_number > 10) {
        return;
    }
    switch (age_) {
    case 45:
        age_--;
        break;
    default: {
        age_++;
    }
}

void Cowboy::makeBang(const int &how_many) {
    for (int i = 0; i < how_many; ++i) {
        std::cout << "Bang!" << std::endl;
    }
}

void someVeryLongMethod(int param1, int param2, int param3, int param4, int param5, int param6, int param7, int param8,
                        int param9, int param10) {
    std::cout << "So long..." << std::endl;
}
} // namespace western

using namespace western;

int main() {
    Cowboy amy;
    Cowboy angus(2);
    Cowboy *current_cowboy = &amy;
    current_cowboy->shoot("yourself");
    return 0;
}

// ADDITIONAL NOTES:
// - LLVM uses spaces, not tabs
// - Every source file should have a header that describes the purpose of it.
// - LLVM prescribes the use of doxygen.
// - #include <iostream> is forbidden and std::endl to be avoided.
