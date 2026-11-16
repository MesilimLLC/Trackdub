using Trackdub.Contracts.Translation;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private static bool IsEligibleModel(TranslationModelDescriptor model)
    {
        // This allowlist is the catalog's explicit commercial license policy. Metadata without evidence is rejected.
        if (model.License is not ("apache-2.0" or "mit" or "cc-by-4.0")) return false;
        if (string.IsNullOrWhiteSpace(model.LicenseEvidence)) return false;
        if (!IsValidRevision(model.Revision)) return false;
        if (model.Files.IsEmpty) return false;
        if (!IsAllowedRepository(model.Repository)) return false;
        return model.Files.All(IsSafeFile);
    }

    private static bool IsValidRevision(string revision) =>
        revision.Length == 40 && revision.All(Uri.IsHexDigit);

    private static bool IsAllowedRepository(string repository)
    {
        if (!repository.StartsWith("onnx-community/opus-mt-", StringComparison.Ordinal) &&
            !repository.StartsWith("Xenova/opus-mt-", StringComparison.Ordinal)) return false;
        if (repository.Split('/').Length != 2) return false;
        return repository.All(character => char.IsAsciiLetterOrDigit(character) || character is '/' or '-' or '_' or '.');
    }

    private static bool IsSafeFile(TranslationModelFile file)
    {
        if (file.SizeBytes < 0) return false;
        if (file.Sha256.Length != 64) return false;
        if (!file.Sha256.All(Uri.IsHexDigit)) return false;
        if (Path.IsPathRooted(file.Path)) return false;
        if (file.Path.Contains('\\')) return false;
        if (file.Path.Contains(':')) return false;
        if (file.Path.Split('/').Any(segment => segment is ".." or ".")) return false;
        return true;
    }

    private static bool HasRequiredFiles(TranslationModelDescriptor model, IReadOnlyList<string> required) =>
        required.All(path => model.Files.Any(file => file.Path == path));
}
