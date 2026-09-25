using System.Text.Json;
using TrainDosy;
if(args.Length<2 || args.Length>3) {Console.Error.WriteLine("Usage: TrainDosy.Cli input.json output.json [http://127.0.0.1:8765/]");return 2;}
try {
    using var cancel=new CancellationTokenSource(TimeSpan.FromMinutes(12));
    Console.CancelKeyPress+=(_,e)=>{e.Cancel=true;cancel.Cancel();};
    var request=JsonSerializer.Deserialize<FitRequest>(await File.ReadAllTextAsync(args[0],cancel.Token),DosyClient.JsonOptions)
        ?? throw new InvalidDataException("Empty request");
    using var http=new HttpClient {BaseAddress=new Uri((args.Length==3?args[2]:"http://127.0.0.1:8765").TrimEnd('/')+"/"),Timeout=TimeSpan.FromMinutes(11)};
    var result=await new DosyClient(http).FitAsync(request,cancel.Token);
    await File.WriteAllTextAsync(args[1],JsonSerializer.Serialize(result,DosyClient.JsonOptions),cancel.Token);
    Console.WriteLine($"{result.Method}: rank={result.SelectedRank}; success={result.Success}; bins={result.X.Length}");
    return result.Success?0:3;
} catch(Exception e) {Console.Error.WriteLine(e.Message);return 1;}
