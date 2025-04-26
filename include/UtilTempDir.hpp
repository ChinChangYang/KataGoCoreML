#pragma once

#include <filesystem>
#include <random>
#include <string>
#include <stdexcept>
#include <system_error>

namespace KataGoCoreML
{

    class TempDir
    {
    public:
        /// Constructs a temp directory with given prefix (default "tmp").
        /// Throws std::runtime_error on failure.
        explicit TempDir(const std::string &prefix = "tmp");

        /// Removes the directory and its contents on destruction.
        ~TempDir();

        /// Returns the path to the created temporary directory.
        const std::filesystem::path &path() const noexcept;

    private:
        std::filesystem::path dir_;

        /// Generates a random alphanumeric suffix of given length.
        static std::string random_suffix(std::size_t len = 6);

        /// Creates a unique directory under the system temp directory.
        /// Throws std::runtime_error on failure.
        static std::filesystem::path make_temp_dir(const std::string &prefix);
    };

    // Implementation

    inline std::string TempDir::random_suffix(std::size_t len)
    {
        static constexpr char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        thread_local std::mt19937_64 eng{std::random_device{}()};
        thread_local std::uniform_int_distribution<std::size_t> dist{0, sizeof(charset) - 2};

        std::string s;
        s.reserve(len);
        for (std::size_t i = 0; i < len; ++i)
        {
            s += charset[dist(eng)];
        }
        return s;
    }

    inline std::filesystem::path TempDir::make_temp_dir(const std::string &prefix)
    {
        namespace fs = std::filesystem;
        fs::path base = fs::temp_directory_path();
        for (int attempt = 0; attempt < 100; ++attempt)
        {
            std::string name = prefix + "_" + random_suffix();
            fs::path candidate = base / name;
            if (!fs::exists(candidate) && fs::create_directory(candidate))
            {
                return candidate;
            }
        }
        throw std::runtime_error("Failed to create unique temp directory");
    }

    inline TempDir::TempDir(const std::string &prefix)
        : dir_(make_temp_dir(prefix))
    {
    }

    inline TempDir::~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    inline const std::filesystem::path &TempDir::path() const noexcept
    {
        return dir_;
    }

} // namespace KataGoCoreML
