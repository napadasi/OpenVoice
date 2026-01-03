import Foundation

public enum TextProcessing {
    public static func splitSentences(text: String, minLength: Int = 10, languageTag: String = "[EN]") -> [String] {
        if languageTag == "EN" || languageTag == "[EN]" {
            return splitSentencesLatin(text: text, minLength: minLength)
        }
        return splitSentencesZh(text: text, minLength: minLength)
    }

    public static func getText(_ text: String, config: OpenVoiceConfig, isSymbol: Bool) -> [Int64] {
        let cleaners = isSymbol ? [] : config.data.text_cleaners
        let cleaned = cleanText(text, cleanerNames: cleaners)
        var sequence = textToSequence(cleaned, symbols: config.symbols ?? [])
        if config.data.add_blank {
            sequence = intersperse(sequence, separator: 0)
        }
        return sequence
    }

    public static func audioConcat(segments: [[Float]], sampleRate: Int, speed: Float = 1.0) -> [Float] {
        var output: [Float] = []
        let silenceCount = Int(Float(sampleRate) * 0.05 / speed)
        let silence = Array(repeating: Float.zero, count: silenceCount)
        for segment in segments {
            output.append(contentsOf: segment)
            output.append(contentsOf: silence)
        }
        return output
    }

    private static func textToSequence(_ text: String, symbols: [String]) -> [Int64] {
        var symbolToId: [String: Int] = [:]
        for (idx, symbol) in symbols.enumerated() {
            symbolToId[symbol] = idx
        }
        var sequence: [Int64] = []
        for scalar in text {
            let symbol = String(scalar)
            if let id = symbolToId[symbol] {
                sequence.append(Int64(id))
            }
        }
        return sequence
    }

    private static func cleanText(_ text: String, cleanerNames: [String]) -> String {
        var output = text
        for cleaner in cleanerNames {
            switch cleaner {
            case "lowercase":
                output = output.lowercased()
            case "collapse_whitespace":
                output = collapseWhitespace(output)
            case "basic_cleaners", "english_cleaners", "transliteration_cleaners":
                output = output.lowercased()
                output = collapseWhitespace(output)
            default:
                continue
            }
        }
        return output
    }

    private static func collapseWhitespace(_ text: String) -> String {
        return text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
    }

    private static func intersperse(_ sequence: [Int64], separator: Int64) -> [Int64] {
        guard !sequence.isEmpty else { return [] }
        var output: [Int64] = []
        output.reserveCapacity(sequence.count * 2 - 1)
        for (index, value) in sequence.enumerated() {
            if index > 0 {
                output.append(separator)
            }
            output.append(value)
        }
        return output
    }

    private static func splitSentencesLatin(text: String, minLength: Int) -> [String] {
        var cleaned = text
        cleaned = cleaned.replacingOccurrences(of: "[。！？；]", with: ".", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "[，]", with: ",", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "[“”]", with: "\"", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "[‘’]", with: "'", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "[\\<\\>\\(\\)\\[\\]\"«»]+", with: "", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "[\\n\\t ]+", with: " ", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "([,.!?;])", with: "$1 $#!", options: .regularExpression)

        var sentences = cleaned.split(separator: "$#!").map { $0.trimmingCharacters(in: .whitespaces) }
        if let last = sentences.last, last.isEmpty {
            sentences.removeLast()
        }

        var result: [String] = []
        var current: [String] = []
        var wordCount = 0
        for (index, sentence) in sentences.enumerated() {
            current.append(sentence)
            wordCount += sentence.split(separator: " ").count
            if wordCount > minLength || index == sentences.count - 1 {
                result.append(current.joined(separator: " "))
                current.removeAll()
                wordCount = 0
            }
        }
        return mergeShortSentencesLatin(result)
    }

    private static func mergeShortSentencesLatin(_ sentences: [String]) -> [String] {
        var output: [String] = []
        for sentence in sentences {
            if let last = output.last, last.split(separator: " ").count <= 2 {
                output[output.count - 1] = last + " " + sentence
            } else {
                output.append(sentence)
            }
        }
        if output.count >= 2, let last = output.last, last.split(separator: " ").count <= 2 {
            output[output.count - 2] += " " + last
            output.removeLast()
        }
        return output
    }

    private static func splitSentencesZh(text: String, minLength: Int) -> [String] {
        var cleaned = text
        cleaned = cleaned.replacingOccurrences(of: "[。！？；]", with: ".", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "[，]", with: ",", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "[\\n\\t ]+", with: " ", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: "([,.!?;])", with: "$1 $#!", options: .regularExpression)

        var sentences = cleaned.split(separator: "$#!").map { $0.trimmingCharacters(in: .whitespaces) }
        if let last = sentences.last, last.isEmpty {
            sentences.removeLast()
        }

        var result: [String] = []
        var current: [String] = []
        var length = 0
        for (index, sentence) in sentences.enumerated() {
            current.append(sentence)
            length += sentence.count
            if length > minLength || index == sentences.count - 1 {
                result.append(current.joined(separator: " "))
                current.removeAll()
                length = 0
            }
        }
        return mergeShortSentencesZh(result)
    }

    private static func mergeShortSentencesZh(_ sentences: [String]) -> [String] {
        var output: [String] = []
        for sentence in sentences {
            if let last = output.last, last.count <= 2 {
                output[output.count - 1] = last + " " + sentence
            } else {
                output.append(sentence)
            }
        }
        if output.count >= 2, let last = output.last, last.count <= 2 {
            output[output.count - 2] += " " + last
            output.removeLast()
        }
        return output
    }
}
