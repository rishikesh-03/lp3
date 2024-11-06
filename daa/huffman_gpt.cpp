#include <bits/stdc++.h>
using namespace std;

struct Node
{
    char ch;
    int freq;
    Node *left, *right;

    // Constructor
    Node(char ch, int freq, Node *left = nullptr, Node *right = nullptr)
        : ch(ch), freq(freq), left(left), right(right) {}

    // Destructor to avoid memory leaks by recursively deleting nodes
    ~Node()
    {
        delete left;
        delete right;
    }
};

// Comparator for the priority queue
struct comp
{
    bool operator()(Node *l, Node *r)
    {
        return l->freq > r->freq;
    }
};

// Function to encode characters by traversing the Huffman Tree
void encode(Node *root, string str, unordered_map<char, string> &huffmanCode)
{
    if (root == nullptr)
        return;

    // If this is a leaf node, it contains a character
    if (!root->left && !root->right)
    {
        huffmanCode[root->ch] = str;
    }

    encode(root->left, str + "0", huffmanCode);
    encode(root->right, str + "1", huffmanCode);
}

// Function to decode the encoded string based on the Huffman Tree
string decode(Node *root, int &index, string str)
{
    string ans="";
    while (index < str.size())
    {
        Node*temp=root;
        while(true)
        {
            if(temp->right==NULL && temp->left==NULL)
            {
                ans+=temp->ch;
                break;

            }
            if(str[index]=='0')
            {
                index++;
                temp=temp->left;
            }
            else{
                index++;
                temp=temp->right;
            }
        }
    }
    return ans;
}

// Function to build the Huffman Tree and perform encoding and decoding
void buildHuffmanTree(string text)
{
    // Edge case: if the text is empty
    if (text.empty())
    {
        cout << "Empty string provided. Nothing to encode or decode.\n";
        return;
    }

    // Count frequency of each character
    unordered_map<char, int> freq;
    for (char ch : text)
    {
        freq[ch]++;
    }

    // Create a priority queue to store live nodes of the Huffman tree
    priority_queue<Node *, vector<Node *>, comp> pq;

    // Create a leaf node for each character and add it to the priority queue
    for (auto pair : freq)
    {
        pq.push(new Node(pair.first, pair.second));
    }

    // Special case: if there is only one unique character
    if (pq.size() == 1)
    {
        cout << "Only one unique character. Encoding is trivial.\n";
        Node *root = pq.top();
        unordered_map<char, string> huffmanCode;
        huffmanCode[root->ch] = "0"; // Assign "0" to this single character

        // Output the encoded and decoded results
        cout << "Huffman Codes are :\n";
        cout << root->ch << " " << huffmanCode[root->ch] << '\n';

        string encodedStr;
        for (char ch : text)
        {
            encodedStr += huffmanCode[ch];
        }

        cout << "\nEncoded string is :\n"
             << encodedStr << '\n';
        cout << "\nDecoded string is: \n"
             << text << '\n';

        delete root; // Clean up memory
        return;
    }

    // Build the Huffman Tree
    while (pq.size() != 1)
    {
        Node *left = pq.top();
        pq.pop();
        Node *right = pq.top();
        pq.pop();

        int sum = left->freq + right->freq;
        pq.push(new Node('\0', sum, left, right));
    }

    Node *root = pq.top();

    // Traverse the Huffman Tree and store Huffman Codes in a map
    unordered_map<char, string> huffmanCode;
    encode(root, "", huffmanCode);

    // Display the Huffman codes
    cout << "Huffman Codes are :\n";
    for (auto pair : huffmanCode)
    {
        cout << pair.first << " " << pair.second << '\n';
    }

    // Encode the input text
    string encodedStr;
    for (char ch : text)
    {
        encodedStr += huffmanCode[ch];
    }

    cout << "\nOriginal string was :\n"
         << text << '\n';
    cout << "\nEncoded string is :\n"
         << encodedStr << '\n';

    // Decode the encoded string
    // int index = -1;
    // cout << "\nDecoded string is: \n";
    // while (index < (int)encodedStr.size() - 2)
    // {
    //     decode(root, index, encodedStr);
    // }

    int index = 0;
    cout << "\nDecoded string is: \n";
    string ans=decode(root,index,encodedStr);
    
    cout<<ans<<endl;

    // Clean up memory
    delete root;
}

int main()
{
    string text = "Huffman coding is a data compression algorithm.";
    buildHuffmanTree(text);

    return 0;
}
