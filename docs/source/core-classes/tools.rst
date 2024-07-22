.. _tools-functions:

Tools Methods
-------------

A set of useful tools in constructing the framework.

.. cpp:class:: tools

   .. cpp:function:: tools()

   .. cpp:function:: ~tools()

   .. cpp:function:: void create_path(std::string path)

   .. cpp:function:: void delete_path(std::string path)

   .. cpp:function:: bool is_file(std::string path)

   .. cpp:function:: std::string absolute_path(std::string path)

   .. cpp:function:: std::vector<std::string> ls(std::string path, std::string ext = "")

   .. cpp:function:: std::string to_string(double val)

   .. cpp:function:: void replace(std::string* in, std::string repl_str, std::string repl_with)

   .. cpp:function:: bool has_string(std::string* inpt, std::string trg)

   .. cpp:function:: bool ends_with(std::string* inpt, std::string val)

   .. cpp:function:: bool has_value(std::vector<std::string>* data, std::string trg)

   .. cpp:function:: std::vector<std::string> split(std::string in, std::string delim)

   .. cpp:function:: std::vector<std::string> split(std::string in, int n)

   .. cpp:function:: std::string hash(std::string input, int len = 18)

   .. cpp:function:: std::string lower(std::string*)
 
   .. cpp:function:: template <typename G> \
                     std::vector<std::vector<G>> discretize(std::vector<G>* v, int N)

   .. cpp:function:: template <typename g> \
                     g max(std::vector<g>* inpt)

   .. cpp:function:: template <typename g> \
                     g min(std::vector<g>* inpt) 



