using System.Globalization;
using NumFlat;
using NumFlat.Clustering;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Probabilistic;
using AiDotNet.Tensors.LinearAlgebra;

const string IrisCsvUrl = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv";
const int ComponentCount = 3;
const int Seed = 42;

var iris = await LoadIrisAsync();
Console.WriteLine($"Loaded Iris: {iris.Length} rows x {iris[0].Length} features from {IrisCsvUrl}");

var numFlat = FitNumFlat(iris);
Console.WriteLine("\nNumFlat GMM fit succeeded.");
PrintParameters(numFlat);

var aiDotNet = FitAiDotNet(iris);
Console.WriteLine("\nAiDotNet GMM fit succeeded.");
PrintParameters(aiDotNet);

var comparison = Compare(numFlat, aiDotNet);
Console.WriteLine("\nBest component matching (NumFlat -> AiDotNet): " + string.Join(", ", comparison.Permutation.Select((ai, nf) => $"{nf}->{ai}")));
Console.WriteLine($"Mean RMSE:       {comparison.MeanRmse:F6}");
Console.WriteLine($"Covariance RMSE: {comparison.CovarianceRmse:F6}");
Console.WriteLine($"Weight MAE:      {comparison.WeightMae:F6}");

static async Task<double[][]> LoadIrisAsync()
{
    using var client = new HttpClient();
    var csv = await client.GetStringAsync(IrisCsvUrl);
    return csv.Split('\n', StringSplitOptions.RemoveEmptyEntries)
        .Skip(1)
        .Select(line => line.Split(','))
        .Where(parts => parts.Length >= 5)
        .Select(parts => parts.Take(4).Select(p => double.Parse(p, CultureInfo.InvariantCulture)).ToArray())
        .ToArray();
}

static GmmParameters FitNumFlat(double[][] data)
{
    var vectors = data.Select(row => new Vec<double>(row.ToArray())).ToArray();
    var options = new GaussianMixtureModelOptions
    {
        Regularization = 1.0e-6,
        MaxIterations = 300,
        Tolerance = 1.0e-8
    };

    var model = Clustering.ToGmm(vectors, ComponentCount, options, new Random(Seed));
    var components = model.Components
        .Select(c => new GmmComponent(
            c.Weight,
            c.Gaussian.Mean.Select(x => x).ToArray(),
            ToArray(c.Gaussian.Covariance)))
        .ToArray();

    return new GmmParameters("NumFlat", components);
}

static GmmParameters FitAiDotNet(double[][] data)
{
    var features = new double[data.Length, data[0].Length];
    for (var i = 0; i < data.Length; i++)
    for (var j = 0; j < data[i].Length; j++)
        features[i, j] = data[i][j];

    var matrix = new Matrix<double>(features);
    var labels = new Vector<double>(new double[data.Length]);
    var options = new GMMOptions<double>
    {
        NumComponents = ComponentCount,
        CovarianceType = CovarianceType.Full,
        RegularizationCovariance = 1.0e-6,
        InitMethod = GMMInitMethod.KMeans,
        MaxIterations = 300,
        Tolerance = 1.0e-8,
        NumInitializations = 1,
        ComputeLowerBound = true,
        AllowLowWeights = true
    };

    using var model = new GaussianMixtureModel<double>(options);
    model.Train(matrix, labels);

    var weights = model.Weights ?? throw new InvalidOperationException("AiDotNet did not expose fitted component weights.");
    var means = model.Means ?? throw new InvalidOperationException("AiDotNet did not expose fitted component means.");
    var covariances = model.Covariances ?? throw new InvalidOperationException("AiDotNet did not expose fitted component covariances.");
    var components = Enumerable.Range(0, ComponentCount)
        .Select(k => new GmmComponent(
            weights[k],
            Enumerable.Range(0, data[0].Length).Select(j => means[k, j]).ToArray(),
            ExtractCovariance(covariances, k, data[0].Length)))
        .ToArray();

    return new GmmParameters("AiDotNet", components);
}

static double[,] ExtractCovariance(Tensor<double> covariances, int component, int dimension)
{
    var covariance = new double[dimension, dimension];
    for (var row = 0; row < dimension; row++)
    for (var col = 0; col < dimension; col++)
        covariance[row, col] = covariances[new[] { component, row, col }];
    return covariance;
}

static double[,] ToArray(Mat<double> matrix)
{
    var array = new double[matrix.RowCount, matrix.ColCount];
    for (var row = 0; row < matrix.RowCount; row++)
    for (var col = 0; col < matrix.ColCount; col++)
        array[row, col] = matrix[row, col];
    return array;
}

static ComparisonResult Compare(GmmParameters left, GmmParameters right)
{
    var bestPermutation = Array.Empty<int>();
    var bestScore = double.PositiveInfinity;
    foreach (var permutation in Permutations(Enumerable.Range(0, right.Components.Length).ToArray()))
    {
        var score = left.Components.Select((component, i) => SquaredDistance(component.Mean, right.Components[permutation[i]].Mean)).Sum();
        if (score < bestScore)
        {
            bestScore = score;
            bestPermutation = permutation.ToArray();
        }
    }

    var meanSquaredErrors = new List<double>();
    var covarianceSquaredErrors = new List<double>();
    var weightAbsoluteErrors = new List<double>();
    for (var i = 0; i < left.Components.Length; i++)
    {
        var a = left.Components[i];
        var b = right.Components[bestPermutation[i]];
        meanSquaredErrors.AddRange(a.Mean.Zip(b.Mean, (x, y) => Math.Pow(x - y, 2.0)));
        weightAbsoluteErrors.Add(Math.Abs(a.Weight - b.Weight));
        for (var row = 0; row < a.Covariance.GetLength(0); row++)
        for (var col = 0; col < a.Covariance.GetLength(1); col++)
            covarianceSquaredErrors.Add(Math.Pow(a.Covariance[row, col] - b.Covariance[row, col], 2.0));
    }

    return new ComparisonResult(
        bestPermutation,
        Math.Sqrt(meanSquaredErrors.Average()),
        Math.Sqrt(covarianceSquaredErrors.Average()),
        weightAbsoluteErrors.Average());
}

static double SquaredDistance(double[] left, double[] right) => left.Zip(right, (x, y) => Math.Pow(x - y, 2.0)).Sum();

static IEnumerable<int[]> Permutations(int[] values)
{
    if (values.Length == 1)
    {
        yield return values;
        yield break;
    }

    for (var i = 0; i < values.Length; i++)
    {
        var head = values[i];
        var tail = values.Where((_, index) => index != i).ToArray();
        foreach (var permutation in Permutations(tail))
            yield return new[] { head }.Concat(permutation).ToArray();
    }
}

static void PrintParameters(GmmParameters parameters)
{
    for (var i = 0; i < parameters.Components.Length; i++)
    {
        var component = parameters.Components[i];
        Console.WriteLine($"  Component {i}: weight={component.Weight:F6}, mean=[{string.Join(", ", component.Mean.Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");
        Console.WriteLine($"    covariance diagonal=[{string.Join(", ", Enumerable.Range(0, component.Covariance.GetLength(0)).Select(d => component.Covariance[d, d].ToString("F6", CultureInfo.InvariantCulture)))}]");
    }
}

sealed record GmmParameters(string Library, GmmComponent[] Components);
sealed record GmmComponent(double Weight, double[] Mean, double[,] Covariance);
sealed record ComparisonResult(int[] Permutation, double MeanRmse, double CovarianceRmse, double WeightMae);
