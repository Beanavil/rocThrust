#ifndef ROCBENCH_CPP14_COMPAT_H
#define ROCBENCH_CPP14_COMPAT_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <memory>

namespace std {
    
    class any {
    public:
        any() : ptr(nullptr) {}

        template<typename T>
        any(T value) : ptr(new Holder<typename std::decay<T>::type>(std::forward<T>(value))) {}

        any(const any& other) : ptr(other.ptr ? other.ptr->clone() : nullptr) {}

        any(any&& other) noexcept : ptr(std::move(other.ptr)) {
            other.ptr = nullptr;
        }

        any& operator=(const any& other) {
            if (this != &other) {
                ptr.reset(other.ptr ? other.ptr->clone() : nullptr);
            }
            return *this;
        }

        any& operator=(any&& other) noexcept {
            if (this != &other) {
                ptr = std::move(other.ptr);
                other.ptr = nullptr;
            }
            return *this;
        }

        template<typename T>
        any& operator=(T&& value) {
            ptr.reset(new Holder<typename std::decay<T>::type>(std::forward<T>(value)));
            return *this;
        }

        bool has_value() const {
            return ptr != nullptr;
        }

        const std::type_info& type() const {
            return ptr ? ptr->type() : typeid(void);
        }

        void reset() {
            ptr.reset();
        }

        friend std::ostream& operator<<(std::ostream& os, const any& any);

    private:
        struct Base {
            virtual ~Base() = default;
            virtual const std::type_info& type() const = 0;
            virtual Base* clone() const = 0;
        };

        template<typename T>
        struct Holder : Base {
            T value;

            Holder(T v) : value(v) {}

            const std::type_info& type() const override {
                return typeid(T);
            }

            Base* clone() const override {
                return new Holder(value);
            }
        };

        std::unique_ptr<Base> ptr;

        template<typename T>
        friend T any_cast(const any& any);
    };

    template<typename T>
    T any_cast(const any& any) {
        if (any.ptr == nullptr || any.type() != typeid(T)) {
            throw std::bad_cast();
        }
        return static_cast<any::Holder<T>*>(any.ptr.get())->value;
    }

    std::ostream& operator<<(std::ostream& os, const any& any) {
        if (any.ptr) {
            if (any.type() == typeid(int)) {
                os << any_cast<int>(any);
            } else if (any.type() == typeid(double)) {
                os << any_cast<double>(any);
            } else if (any.type() == typeid(std::string)) {
                os << any_cast<std::string>(any);
            } else {
                os << "Unknown type";
            }
        } else {
            os << "none";
        }
        return os;
    }

}

#endif // ROCBENCH_CPP14_COMPAT_H