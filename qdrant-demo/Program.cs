using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Connectors.Memory.Qdrant;


var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", true, true)
    .AddUserSecrets<Program>()
    .Build();

string? myAzureOpenAIEmbedDeployment = config["AzureOpenAIEmbedDeployment"];
string? myAzureOpenAIDeployment = config["AzureOpenAIDeployment"];
string? myAOAIEndpoint = config["AzureOpenAIEndpoint"];
string? myAOAIKey = config["AzureOpenAIKey"];
string? myQdrantEndpoint = config["QdrantEndpoint"];

Console.WriteLine(myAOAIEndpoint);
Console.WriteLine(myAOAIKey);

//string MemoryCollectionName = "qdrant-test";
int qdrantVectorSize = 1536;
QdrantMemoryStore memoryStore = new(myQdrantEndpoint!, qdrantVectorSize);


IKernel kernel = Kernel.Builder
            // .WithLogger(ConsoleLogger.Log)
            .WithAzureTextCompletionService(myAzureOpenAIDeployment!, myAOAIEndpoint!, myAOAIKey!)
            .WithAzureTextEmbeddingGenerationService(myAzureOpenAIEmbedDeployment!, myAOAIEndpoint!, myAOAIKey!)
            .WithMemoryStorage(memoryStore)
            //.WithMemoryStorage(new VolatileMemoryStore())
            //.WithQdrantMemoryStore(Env.Var("QDRANT_ENDPOINT"), 1536) // This method offers an alternative approach to registering Qdrant memory store.
            .Build();

const string memoryCollectionName = "Facts About Me";

var myBio = await kernel.Memory.SaveInformationAsync(memoryCollectionName, id: "LinkedIn Bio",
    text: "I currently work in the hotel industry at the front desk. I won the best team player award.");

var myHistory = await kernel.Memory.SaveInformationAsync(memoryCollectionName, id: "LinkedIn History",
    text: "I have worked as a tourist operator for 8 years. I have also worked as a banking associate for 3 years.");

var myRecentFB = await kernel.Memory.SaveInformationAsync(memoryCollectionName, id: "Recent Facebook Post",
    text: "My new dog Trixie is the cutest thing you've ever seen. She's just 2 years old.");

var myOldFB = await kernel.Memory.SaveInformationAsync(memoryCollectionName, id: "Old Facebook Post",
    text: "Can you believe the size of the trees in Yellowstone? They're huge! I'm so committed to forestry concerns.");

Console.WriteLine("Four GIGANTIC vectors were generated just now from those 4 pieces of text above.");

// loop thru all the memories in the collection

Console.WriteLine("== Printing Collections in DB ==");
var printColl = memoryStore.GetCollectionsAsync();
await foreach (var collection in printColl)
{
    Console.WriteLine(collection);
}

Console.WriteLine("== Retrieving Memories Through the Kernel ==");
MemoryQueryResult? lookup = await kernel.Memory.GetAsync(memoryCollectionName, "LinkedIn Bio");
Console.WriteLine(lookup != null ? lookup.Metadata.Text : "ERROR: memory not found");

Console.WriteLine("== Retrieving Memories Directly From the Store ==");
var memory1 = await memoryStore.GetWithPointIdAsync(memoryCollectionName, myOldFB);
Console.WriteLine(memory1 != null ? memory1.Metadata.Text : "ERROR: memory not found");

Console.WriteLine("== Similarity Searching Memories: My favorite work is ==");
var searchResults = kernel.Memory.SearchAsync(memoryCollectionName, "My work information", limit: 3, minRelevanceScore: 0.8);

var relatedMemory = "I know nothing.";
var counter = 0;

await foreach (var item in searchResults)
{
    if (counter == 0) { relatedMemory = item.Metadata.Text; }
    Console.WriteLine(item.Metadata.Text + " : " + item.Relevance);        
}

var myFunction = kernel.CreateSemanticFunction(@"
{{$input}}
Tell me about me and my work history in less than 70 characters.
", maxTokens: 100, temperature: 0.1, topP: .1);

var result = await myFunction.InvokeAsync(relatedMemory);

Console.WriteLine("Response from AI: " +result);


//Add test collection and delete it
const string memoryCollectionDelete = "Memory to Delete";

var delMoon = await kernel.Memory.SaveInformationAsync(memoryCollectionDelete, id: "Moon distance",
    text: "The average distance is 238,855 miles (384,400 kilometers)");
var delParis = await kernel.Memory.SaveInformationAsync(memoryCollectionDelete, id: "Paris NYC History",
    text: "The distance between Paris and New York City is approximately 3,631 miles");


Console.WriteLine("== Printing Collections in DB ==");
await foreach (var collection in printColl)
{
    Console.WriteLine(collection);
}

Console.WriteLine("== Removing Collection {0} ==", memoryCollectionDelete);
await memoryStore.DeleteCollectionAsync(memoryCollectionDelete);

Console.WriteLine("== Printing Collections in DB ==");
await foreach (var collection in printColl)
{
    Console.WriteLine(collection);
}
