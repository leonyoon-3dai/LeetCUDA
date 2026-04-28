// 08 · Move Semantics & RVO — C++ 면접 절대 단골
//
// 질문 단골 4개:
//   1) "복사 생성자, 이동 생성자, 대입의 차이?"
//   2) "RVO/NRVO가 뭐고 어떻게 동작하나?"
//   3) "std::move는 실제로 뭘 하나? (캐스트일 뿐인가?)"
//   4) "rule of 0/3/5는 무엇이며 언제 따라야 하나?"
//
// 이 데모는 인스트루먼트된 클래스로 복사/이동 횟수를 출력해 답을 시각화합니다.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

struct Counters {
    int constructed = 0;
    int copies      = 0;
    int moves       = 0;
    int destroyed   = 0;
    void reset() { *this = {}; }
    void print(const char* tag) const {
        std::printf("  [%s] ctor=%d  copy=%d  move=%d  dtor=%d\n",
                    tag, constructed, copies, moves, destroyed);
    }
};
static Counters cnt;

// 큰 버퍼를 들고 있어서 복사/이동 비용이 분명한 RAII 객체
class Buffer {
public:
    Buffer(size_t n, char fill) : size_(n), data_(new char[n]) {
        std::memset(data_, fill, n);
        ++cnt.constructed;
    }
    // 복사 — 새 메모리 할당 + memcpy (비싸다)
    Buffer(const Buffer& o) : size_(o.size_), data_(new char[o.size_]) {
        std::memcpy(data_, o.data_, size_);
        ++cnt.copies;
    }
    // 이동 — 포인터만 훔침 (싸다, O(1))
    Buffer(Buffer&& o) noexcept : size_(o.size_), data_(o.data_) {
        o.size_ = 0; o.data_ = nullptr;
        ++cnt.moves;
    }
    Buffer& operator=(const Buffer& o) {
        if (this == &o) return *this;
        delete[] data_;
        size_ = o.size_;
        data_ = new char[size_];
        std::memcpy(data_, o.data_, size_);
        ++cnt.copies;
        return *this;
    }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this == &o) return *this;
        delete[] data_;
        size_ = o.size_;  data_ = o.data_;
        o.size_ = 0;      o.data_ = nullptr;
        ++cnt.moves;
        return *this;
    }
    ~Buffer() { delete[] data_; ++cnt.destroyed; }

    size_t size() const { return size_; }
private:
    size_t size_;
    char*  data_;
};

// (1) 함수에서 값으로 반환 — 컴파일러가 RVO/NRVO로 복사/이동을 elision함
//     C++17 이후 prvalue copy elision은 거의 강제 (guaranteed copy elision).
Buffer make_buffer_RVO() {
    return Buffer(1 << 20, 'A');   // prvalue로 즉시 반환 객체 자리에 만든다
}

// (2) NRVO — named local을 반환. 컴파일러가 가능하면 elision, 아니면 move.
Buffer make_buffer_NRVO() {
    Buffer b(1 << 20, 'B');
    return b;   // 가능하면 elision. 분기가 있거나 두 변수면 move로 떨어짐.
}

// (3) 분기 있는 NRVO — 보통 elision 안 됨, move로 fallback
Buffer make_buffer_branch(bool cond) {
    Buffer x(1 << 20, 'X');
    Buffer y(1 << 20, 'Y');
    if (cond) return x;
    return y;
}

// (4) std::move 의도 — 명시적으로 lvalue를 rvalue로 캐스트
void take(Buffer&&) { /* sink */ }

// 면접용 함수 시그니처: const T&  vs  T&&  vs  T (by value)
// "perfect forwarding"은 템플릿 + std::forward를 써야 함.
template <class T>
void wrapper_perfect(T&& v) {
    take(std::forward<T>(v));   // lvalue면 lvalue로, rvalue면 rvalue로 그대로 전달
}

int main() {
    std::printf("== (1) RVO (prvalue 반환) ==\n");
    cnt.reset();
    {
        Buffer b = make_buffer_RVO();
        (void)b;
    }
    cnt.print("RVO");
    std::printf("  기대: copy=0, move=0 (C++17 guaranteed elision)\n\n");

    std::printf("== (2) NRVO (named local 반환) ==\n");
    cnt.reset();
    {
        Buffer b = make_buffer_NRVO();
        (void)b;
    }
    cnt.print("NRVO");
    std::printf("  기대: copy=0; move 0 또는 1 (컴파일러/플래그에 따라).\n\n");

    std::printf("== (3) 분기 있는 NRVO — fallback to move ==\n");
    cnt.reset();
    {
        Buffer b = make_buffer_branch(true);
        (void)b;
    }
    cnt.print("branch");
    std::printf("  기대: copy=0, move ≥ 1 (컴파일러가 elide 못 하면 move).\n\n");

    std::printf("== (4) lvalue를 sink로 — std::move 없이 vs 있이 ==\n");
    cnt.reset();
    {
        Buffer b(1 << 20, 'L');
        take(std::move(b));  // ★ std::move 없으면 컴파일 에러 (rvalue 참조 sink)
    }
    cnt.print("std::move");
    std::printf("  핵심: std::move는 \"runtime 작업 0\"인 캐스트일 뿐. 실제 이동은\n");
    std::printf("        대상 클래스의 move ctor/op이 한다.\n\n");

    std::printf("== (5) perfect forwarding ==\n");
    cnt.reset();
    {
        Buffer b(1 << 20, 'P');
        wrapper_perfect(std::move(b));   // T&& → rvalue로 forward
    }
    cnt.print("perfect-fwd");

    std::printf("\n포인트 (NVIDIA 면접 답안):\n");
    std::printf("  - Rule of 0: RAII 멤버를 잘 쓰면 직접 구현할 필요 자체가 없음.\n");
    std::printf("  - Rule of 5: 하나라도 직접 구현하면 dtor/copy/move/copy=/move= 모두 필요.\n");
    std::printf("  - move ctor에는 가급적 noexcept — vector 등이 move를 선호하게 만든다.\n");
    std::printf("  - std::move는 캐스트, std::forward는 \"forwarding reference\"를 보존하는 캐스트.\n");
    std::printf("  - RVO/NRVO는 컴파일러가 빈 자리에 직접 객체를 짓는 최적화.\n");
    return 0;
}
