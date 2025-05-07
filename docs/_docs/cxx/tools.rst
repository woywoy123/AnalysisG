.. cpp:class:: tools

    A collection of static utility functions.

    Provides helper functions for common tasks such as file I/O, string processing,
    Base64 encoding/decoding, and generic vector operations like finding min/max,
    summing, splitting, and selecting elements. All methods are static, so no instance
    of the `tools` class needs to be created.

    .. cpp:function:: tools()

        Default constructor for the tools class.

        .. note:: As all methods are static, this constructor typically serves no purpose.

    .. cpp:function:: ~tools()

        Destructor for the tools class.

        .. note:: As all methods are static, this destructor typically serves no purpose.

    .. rubric:: File System Operations (io.cxx)

    .. cpp:function:: static void create_path(std::string path)

        Creates a directory path recursively.

        Given a full path (which can be a file path or directory path), this function
        creates all necessary parent directories. If the input path points to a file,
        the file name part is ignored, and only the directory structure is created.
        Uses `mkdir` with permissions S_IRWXU for each directory level.

        :param path: The full path (file or directory) for which the directory structure should be created.

    .. cpp:function:: static void delete_path(std::string path)

        Deletes a file or an empty directory.

        Checks if the path exists. If it's a directory, `rmdir` is used (fails if not empty).
        If it's a file, `unlink` is used. If the path doesn't exist, nothing happens.

        :param path: The path to the file or directory to be deleted.

    .. cpp:function:: static bool is_file(std::string path)

        Checks if a file or directory exists at the specified path.

        Uses `stat` to determine if an entry exists at the given path.

        :param path: The path to check.
        :return: ``true`` if a file or directory exists at the path, ``false`` otherwise.

    .. cpp:function:: static void rename(std::string start, std::string target)

        Renames or moves a file or directory.

        Wraps `std::filesystem::rename`. Can be used to rename a file/directory
        within the same directory or move it to a different location.

        :param start: The original path of the file or directory.
        :param target: The new path for the file or directory.

    .. cpp:function:: static std::string absolute_path(std::string path)

        Returns the canonical absolute path for a given path.

        Wraps `std::filesystem::canonical`. Resolves symbolic links and "."/".." components
        to produce an absolute path. Throws `std::filesystem::filesystem_error` if the
        path does not exist or resolution fails.

        :param path: The input path (relative or absolute).
        :return: The canonical absolute path as a string.

    .. cpp:function:: static std::vector<std::string> ls(std::string path, std::string ext = "")

        Lists files recursively within a directory, optionally filtering by extension.

        Traverses the specified directory and its subdirectories. Returns a vector
        containing the canonical absolute paths of all files found. If an extension
        is provided (e.g., ".txt"), only files ending with that extension are included.
        Handles potential exceptions during directory iteration. If the input path ends with '*',
        the '*' is removed before processing.

        :param path: The starting directory path for the recursive search. Can end with '*'.
        :param ext: Optional file extension filter (e.g., ".log", ".cpp"). If empty, all files are listed.
        :return: A vector of strings, each containing the absolute path to a found file. Returns an empty vector if the path is invalid or inaccessible.

    .. rubric:: String Manipulation (strings.cxx)

    .. cpp:function:: static std::string to_string(double val)

        Converts a double-precision floating-point number to its string representation.

        Uses a stringstream for the conversion. Default precision is used.

        :param val: The double value to convert.
        :return: The string representation of the double.

    .. cpp:function:: static std::string to_string(double val, int prec)

        Converts a double-precision floating-point number to its string representation with fixed precision.

        Uses a stringstream with `std::fixed` and the specified precision for the conversion.

        :param val: The double value to convert.
        :param prec: The number of digits to display after the decimal point. If negative, default precision might be used (behavior depends on stringstream).
        :return: The string representation of the double with the specified precision.

    .. cpp:function:: static void replace(std::string* in, std::string repl_str, std::string repl_with)

        Replaces all occurrences of a substring within a string (in-place).

        Modifies the input string directly by replacing every instance of `repl_str` with `repl_with`.

        :param in: Pointer to the string to be modified.
        :param repl_str: The substring to search for.
        :param repl_with: The string to replace `repl_str` with.

    .. cpp:function:: static bool has_string(std::string* inpt, std::string trg)

        Checks if a string contains a specific substring.

        Uses `std::string::find` to determine if `trg` exists within the string pointed to by `inpt`.

        :param inpt: Pointer to the string to search within.
        :param trg: The substring to search for.
        :return: ``true`` if `trg` is found within `*inpt`, ``false`` otherwise.

    .. cpp:function:: static bool ends_with(std::string* inpt, std::string val)

        Checks if a string ends with a specific suffix.

        Compares the end portion of the string pointed to by `inpt` with the `val` string.
        Returns `false` if the input string is shorter than the suffix.

        :param inpt: Pointer to the string to check.
        :param val: The suffix string to check for.
        :return: ``true`` if `*inpt` ends with `val`, ``false`` otherwise.

    .. cpp:function:: static bool has_value(std::vector<std::string>* data, std::string trg)

        Checks if a vector of strings contains a specific string value.

        Iterates through the vector and compares each element with the target string.

        :param data: Pointer to the vector of strings to search within.
        :param trg: The string value to search for.
        :return: ``true`` if `trg` is found in the vector, ``false`` otherwise.

    .. cpp:function:: static std::vector<std::string> split(std::string in, std::string delim)

        Splits a string into a vector of substrings based on a delimiter.

        Breaks the input string `in` into pieces wherever the `delim` string occurs.
        The delimiters themselves are not included in the output vector.
        The part of the string after the last delimiter is included as the final element.

        :param in: The string to be split.
        :param delim: The delimiter string used for splitting.
        :return: A vector of strings resulting from the split.

    .. cpp:function:: static std::vector<std::string> split(std::string in, size_t n)

        Splits a string into a vector of substrings of a fixed maximum size.

        Divides the input string `in` into chunks, where each chunk has a maximum length of `n`.
        The last chunk may be shorter if the total string length is not a multiple of `n`.

        :param in: The string to be split.
        :param n: The maximum size of each chunk.
        :return: A vector of strings, each representing a chunk of the original string.

    .. cpp:function:: static std::string hash(std::string input, int len = 18)

        Generates a hexadecimal hash string from an input string.

        Uses `std::hash<std::string>` to compute a hash value, then converts it
        to a hexadecimal string prefixed with "0x". The resulting string is padded
        with trailing zeros or truncated (from the end, which might not be ideal hash behavior)
        to match the desired length `len`.

        :param input: The string to hash.
        :param len: The desired length of the output hash string (including the "0x" prefix). Defaults to 18.
        :return: The generated hexadecimal hash string.

    .. cpp:function:: static std::string lower(std::string*)

        Converts a string to its lowercase equivalent.

        Creates a new string and fills it with the lowercase versions of the characters
        from the input string.

        :param in: Pointer to the string to convert.
        :return: A new string containing the lowercase version of `*in`.

    .. rubric:: Encoding/Decoding (base64.cxx)

    .. cpp:function:: static std::string encode64(std::string* data)

        Encodes a string into Base64 format.

        Takes a string, treats its content as raw bytes, and encodes them using Base64.

        :param data: Pointer to the string to be encoded.
        :return: The Base64 encoded string.

    .. cpp:function:: static std::string encode64(unsigned char const* bytes_to_encode, unsigned int len)

        Encodes a sequence of raw bytes into Base64 format.

        Performs Base64 encoding on the provided byte array.

        :param bytes_to_encode: Pointer to the start of the byte sequence.
        :param len: The number of bytes in the sequence.
        :return: The Base64 encoded string.

    .. cpp:function:: static std::string decode64(std::string* inpt)

        Decodes a Base64 encoded string.

        Takes a string containing Base64 data and decodes it back into its original form.
        Assumes the input string is valid Base64. Ignores invalid characters and padding ('=') position issues.

        :param inpt: Pointer to the Base64 encoded string.
        :return: The decoded string.

    .. cpp:function:: static std::string decode64(std::string const& s)

        Decodes a Base64 encoded string.

        Takes a string containing Base64 data and decodes it back into its original form.
        Assumes the input string is valid Base64. Ignores invalid characters and padding ('=') position issues.

        :param s: The Base64 encoded string (passed by const reference).
        :return: The decoded string.

    .. rubric:: Template Functions

    .. cpp:function:: template <typename G> static std::vector<std::vector<G>> discretize(std::vector<G>* v, int N)

        Discretizes a vector into sub-vectors (chunks) of a specified size.

        Splits the input vector `v` into multiple smaller vectors, each containing
        at most `N` elements. The last sub-vector may contain fewer than `N` elements
        if the total size of `v` is not divisible by `N`.

        :tparam G: The type of elements stored in the vector.
        :param v: Pointer to the input vector to be discretized.
        :param N: The maximum size of each chunk (sub-vector).
        :return: A vector of vectors, where each inner vector is a chunk of the original.

    .. cpp:function:: template <typename g> static g max(std::vector<g>* inpt)

        Finds the maximum element in a vector.

        Iterates through the vector to find the largest element according to the
        less-than-or-equal-to operator (`<=`) defined for type `g`. Assumes the vector is not empty.

        :tparam g: The type of elements in the vector. Must support comparison (`<=`).
        :param inpt: Pointer to the input vector.
        :return: The maximum value found in the vector.

    .. cpp:function:: template <typename g> static g min(std::vector<g>* inpt)

        Finds the minimum element in a vector.

        Iterates through the vector to find the smallest element according to the
        greater-than-or-equal-to operator (`>=`) defined for type `g`. Assumes the vector is not empty.

        :tparam g: The type of elements in the vector. Must support comparison (`>=`).
        :param inpt: Pointer to the input vector.
        :return: The minimum value found in the vector.

    .. cpp:function:: template <typename g> static g sum(std::vector<g>* inpt)

        Calculates the sum of all elements in a vector.

        Iterates through the vector and accumulates the sum of its elements.
        Assumes the type `g` supports the addition assignment operator (`+=`) and
        can be initialized from `0`.

        :tparam g: The type of elements in the vector. Must support `+=` and initialization from 0.
        :param inpt: Pointer to the input vector.
        :return: The sum of all elements in the vector.

    .. cpp:function:: template <typename g> static std::vector<g*> put(std::vector<g*>* src, std::vector<int>* trg)

        Creates a new vector of pointers by selecting elements from a source vector based on indices.

        Constructs a new vector of pointers (`std::vector<g*>`). For each index in the `trg` vector,
        it retrieves the pointer at that index from the `src` vector and adds it to the new vector.
        The size of the output vector will be the same as the size of the `src` vector, but elements
        not selected by `trg` indices will be `nullptr`. This seems potentially incorrect, as it initializes
        the output vector to the size of `src` but only fills based on `trg`. The behavior might be unexpected
        if `trg` contains indices out of bounds for `src`.

        :tparam g: The underlying type pointed to by the elements in the vectors.
        :param src: Pointer to the source vector of pointers (`std::vector<g*>*`).
        :param trg: Pointer to the vector of integer indices (`std::vector<int>*`).
        :return: A new vector containing pointers selected from `src` based on indices in `trg`, with `nullptr` for unselected positions up to the size of `src`.

    .. cpp:function:: template <typename g> static void put(std::vector<g*>* out, std::vector<g*>* src, std::vector<int>* trg)

        Populates an output vector with pointers selected from a source vector based on indices, marking them as 'in_use'.

        Clears the `out` vector, reserves space based on the size of `trg`, and then populates `out`.
        For each index `i` in the `trg` vector, it retrieves the pointer `v` from `(*src)[(*trg)[i]]`,
        pushes `v` onto the `out` vector, and sets `v->in_use = 1`.
        Assumes that type `g` has a member variable `in_use` and that indices in `trg` are valid for `src`.

        :tparam g: The underlying type pointed to by the elements in the vectors. Must have an `in_use` member.
        :param out: Pointer to the output vector of pointers (`std::vector<g*>*`) to be populated.
        :param src: Pointer to the source vector of pointers (`std::vector<g*>*`).
        :param trg: Pointer to the vector of integer indices (`std::vector<int>*`).

    .. cpp:function:: template <typename g> static void unique_key(std::vector<g>* inx, std::vector<g>* oth)

        Adds unique elements from one vector (`inx`) to another vector (`oth`).

        Iterates through the `inx` vector. For each element, it checks if that element
        already exists in the `oth` vector (using a map for efficient lookup). If the element
        is not found in `oth`, it is added to `oth`.

        :tparam g: The type of elements in the vectors. Must be usable as a key in `std::map`.
        :param inx: Pointer to the vector containing elements to potentially add.
        :param oth: Pointer to the vector to which unique elements from `inx` will be added.

