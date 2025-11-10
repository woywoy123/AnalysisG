.. _tools-functions:

Tools Methods
-------------

A set of useful tools in constructing the framework.
To use the class, include the following header in your project.

.. code:: C++

   #include <tools/tools.h>

.. cpp:class:: tools

   .. cpp:function:: tools()

   .. cpp:function:: ~tools()

   .. cpp:function:: void create_path(std::string path)

      Creates a new folder path.

   .. cpp:function:: void delete_path(std::string path)

      Deletes the specifies file directory.

   .. cpp:function:: bool is_file(std::string path)

      Checks if the given path is a file.

   .. cpp:function:: std::string absolute_path(std::string path)

      Returns the absolute path.

   .. cpp:function:: std::vector<std::string> ls(std::string path, std::string ext = "")

      Returns the directory of the given input path, or only returns files with the specified extension.

   .. cpp:function:: std::string to_string(double val)

      Converts the value as a string.

   .. cpp:function:: std::string to_string(double val, int prec)

      Converts the value as a string with given precision.

   .. cpp:function:: void replace(std::string* in, std::string repl_str, std::string repl_with)

   .. cpp:function:: bool has_string(std::string* inpt, std::string trg)

   .. cpp:function:: bool ends_with(std::string* inpt, std::string val)

   .. cpp:function:: bool has_value(std::vector<std::string>* data, std::string trg)

   .. cpp:function:: std::vector<std::string> split(std::string in, std::string delim)

   .. cpp:function:: std::vector<std::string> split(std::string in, int n)

   .. cpp:function:: std::string hash(std::string input, int len = 18)

   .. cpp:function:: std::string lower(std::string*)

   .. cpp:function:: std::string encode64(unsigned char const*, unsigned int len)

   .. cpp:function:: std::string decode64(std::string* inpt)

   .. cpp:function:: std::string decode64(std::string const& s)

   .. cpp:function:: template <typename G> std::vector<std::vector<G>> discretize(std::vector<G>* v, int N)

   .. cpp:function:: template <typename g> g max(std::vector<g>* inpt)

   .. cpp:function:: template <typename g> g min(std::vector<g>* inpt) 

   .. cpp:function:: template <typename g> g sum(std::vector<g>* inpt) 
